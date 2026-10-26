"""The SM120 decode kernels of `mm_abt` for at most 128 token rows. Each work group streams one range of the
(weight tile, K block) units of the schedule of `decode`, with 128 weight rows as the 16-row operand of `mma.sync`
and a tile of `T` token rows as its 8-column operand. `T` is the smallest entry of `TOKEN_TILES` that holds `M`."""

from . import decode, fp8_mm, runtime

ARCH = 120
TOKEN_TILES = (16, 32, 64, 128)
SIGNATURE = "mmppppppuuuuu"


def kernel_name(recipe, token_tile):
    return f"{recipe}_mm_abt_decode_sm{ARCH}_t{token_tile}"


def token_tile(m):
    return next(tile for tile in TOKEN_TILES if tile >= m)


def mm_abt_decode(problem, ctas=None):
    """Launches the decode kernel on `ctas` work groups (the SM count by default) and returns the schedule and the
    partial workspace."""
    capability = runtime.compute_capability(problem.device_index)
    schedule = decode.plan(capability, problem.recipe, problem.m, problem.padded_n, problem.k,
                           ctas or decode.ctas(problem.device_index), token_tile(problem.m))
    partials = decode.workspace(problem, schedule)
    tokens = schedule.token_tile
    a, b, tensors = problem.a, problem.b, problem.epilogue.tensors
    bias, residual = tensors.get("bias"), tensors.get("residual")
    values = (problem.tensor_map("decode_w", b.values, (decode.TILE, decode.BLOCK)),
              problem.tensor_map(f"decode_x{tokens}", a.values, (tokens, decode.BLOCK)),
              problem.outputs["y"].data_ptr(), 0 if bias is None else bias.data_ptr(),
              0 if residual is None else residual.data_ptr(), a.scales.data_ptr(), b.scales.data_ptr(),
              partials.data_ptr(), problem.m, problem.n, problem.padded_n, problem.k, schedule.ctas)
    decode.launch(problem, kernel_name(problem.recipe, tokens), schedule, SIGNATURE, values, partials)
    return schedule, partials


for recipe in fp8_mm.RECIPES:
    decode.register(ARCH, recipe, mm_abt_decode)
