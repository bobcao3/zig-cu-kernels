"""The SM90 decode kernels of `mm_abt` for at most `decode.MAX_M` rows of `A` and the recipes `fp8_b128_e8m0`,
`fp8_b128_fp32` and `fp8_rowwise`. Each kernel computes `Yᵀ = B Aᵀ` with `wgmma`, whose M operand is a tile of 128
weight rows and whose N operand is the token tile `T` of `A`, over the stream-K schedule of `decode.plan`. The
block-scaled recipes multiply the product of each K block by its weight and token scales before they add it to the
sums; the rowwise recipe adds the products unscaled and scales the sums in the epilogue."""

import functools

from . import decode, fp8_mm, runtime
from .packs import ceil_div

ARCH = 90
SIGNATURE = "mmpppppuuuuu"
SCALE_ATOM_BLOCKS = 4
UNIT_BYTES = decode.TILE * decode.BLOCK
STREAM_BYTES_PER_US = 2.605e6
CTA_UNIT_US = 0.30
CTA_TOKEN_US = 0.0026
PARTIAL_BYTES_PER_US = 3.025e6
RECIPE_EPILOGUES = {"fp8_b128_e8m0": ("store",), "fp8_b128_fp32": ("store",), "fp8_rowwise": ("store", "bias")}


def kernel_name(recipe, token_tile, epilogue):
    return f"{recipe}_decode_sm{ARCH}_t{token_tile}" + ("_bias" if epilogue == "bias" else "")


def partial_count(tiles, blocks, ctas):
    units = tiles * blocks
    count = 0
    for cta in range(ctas):
        lo, hi = cta * units // ctas, (cta + 1) * units // ctas
        first, last = lo // blocks, (hi - 1) // blocks
        count += (lo % blocks != 0 or hi < (first + 1) * blocks) + (last != first and hi % blocks != 0)
    return count


@functools.lru_cache(maxsize=256)
def choose_ctas(tiles, blocks, token_tile, sms):
    """The CTA count from 90% of `sms` to `sms` with the least modeled time: the weight stream at the H100 read rate,
    the units of the busiest CTA at the rate one CTA reaches, and the partials of shared tiles written and read once."""
    units = tiles * blocks

    def cost(ctas):
        stream = units * UNIT_BYTES / STREAM_BYTES_PER_US
        busiest = -(-units // ctas) * (CTA_UNIT_US + CTA_TOKEN_US * token_tile)
        partials = partial_count(tiles, blocks, ctas) * 2 * token_tile * decode.TILE * 4 / PARTIAL_BYTES_PER_US
        return max(stream, busiest) + partials

    return min(range(min(sms, units), min(sms, units) * 9 // 10 - 1, -1), key=cost)


def schedule_of(problem, ctas=None):
    capability = runtime.compute_capability(problem.device_index)
    if ctas is None:
        probe = decode.plan(capability, problem.recipe, problem.m, problem.padded_n, problem.k, 1)
        ctas = choose_ctas(probe.tiles, probe.blocks, probe.token_tile, decode.ctas(problem.device_index))
    return decode.plan(capability, problem.recipe, problem.m, problem.padded_n, problem.k, ctas)


def mm_abt_decode(problem: fp8_mm.Problem, ctas=None):
    schedule = schedule_of(problem, ctas)
    assert schedule.units * schedule.ctas < 2**32, "the kernel computes the unit ranges in 32-bit integers"
    token_tile = schedule.token_tile
    partials = decode.workspace(problem, schedule)
    bias = problem.epilogue.tensors.get("bias")
    values = (problem.tensor_map("b_decode", problem.b.values, (decode.TILE, decode.BLOCK)),
              problem.tensor_map(f"a_decode_{token_tile}", problem.a.values, (token_tile, decode.BLOCK)),
              problem.outputs["y"].data_ptr(), partials.data_ptr(), problem.a.scales.data_ptr(),
              problem.b.scales.data_ptr(), 0 if bias is None else bias.data_ptr(), problem.m, problem.n,
              schedule.tiles, schedule.blocks, SCALE_ATOM_BLOCKS * ceil_div(problem.k, SCALE_ATOM_BLOCKS * decode.BLOCK))
    name = kernel_name(problem.recipe, token_tile, problem.epilogue.kind)
    decode.launch(problem, name, schedule, SIGNATURE, values, partials, pdl=True)


for recipe, epilogues in RECIPE_EPILOGUES.items():
    decode.register(ARCH, recipe, mm_abt_decode, epilogues)
