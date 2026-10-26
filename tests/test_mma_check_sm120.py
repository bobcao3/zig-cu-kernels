"""Checks the FP8 `mma.sync` forms and the tensor map copies of compute capability 12.x on single tiles."""

import pytest
import torch

from conftest import cuda_seed, declare_gpu_work
from zig_cu_kernels import runtime, tensormap

F8F6F4 = "mma_check_sm120_f8f6f4"
BLOCK_SCALE = {(0, 0, 0, 0): "mma_check_sm120_block_scale_a00_b00",
               (3, 1, 2, 3): "mma_check_sm120_block_scale_a31_b23"}
TMA_COPY = "mma_check_sm120_tma_copy"
TILE_ROWS, TILE_COLUMNS = 64, 128
M, N, K = 16, 8, 32
# Byte j of the probe scale word of lane l holds PROBE_BASE + 4·l + j.
PROBE_BASE = 64
# Products of E4M3 values are exact in FP32; the bound allows the tensor cores a shorter accumulator.
SUM_TOLERANCE = 2**-14


def kernel(name):
    reason = runtime.unavailable_reason(name)
    if reason is not None:
        pytest.skip(reason)
    return runtime.load(name)[0]


def launch(function, signature, values, threads=32):
    runtime.run(function, (1, 1, 1), threads, 0, signature, values, torch.cuda.current_stream().cuda_stream)


def random_e4m3(rows, columns):
    return torch.randn(rows, columns, device="cuda").to(torch.float8_e4m3fn)


def product(a, b, scale_a=None, scale_b=None):
    a64, b64 = a.double(), b.double()
    if scale_a is not None:
        a64 = a64 * torch.exp2(scale_a.double() - 127)[:, None]
        b64 = b64 * torch.exp2(scale_b.double() - 127)[:, None]
    return a64 @ b64.T, a64.abs() @ b64.abs().T


def check_close(result, expected, magnitude):
    error = (result.double() - expected).abs()
    assert torch.all(error <= SUM_TOLERANCE * magnitude + 1e-30), f"largest error {error.max().item()}"


def run_mma(name, a, b, scale_a_words=None, scale_b_words=None):
    d = torch.full((M, N), float("nan"), device="cuda")
    words = [a.view(torch.int32), b.view(torch.int32)]
    if scale_a_words is not None:
        words += [scale_a_words, scale_b_words]
    launch(kernel(name), "p" * (1 + len(words)), (d.data_ptr(), *(word.data_ptr() for word in words)))
    return d


def expected_lanes(selectors):
    """Returns the lane and byte that supply the scale of each row of A and each column of B (PTX ISA 9.7.16.3):
    the lane pair `thread-id-a` of quad `row % 8` supplies rows `row % 8` and `row % 8 + 8`, and lane `thread-id-b`
    of quad `column` supplies column `column`."""
    a_byte, a_thread, b_byte, b_thread = selectors
    rows = [(4 * (row % 8) + 2 * a_thread + row // 8, a_byte) for row in range(M)]
    columns = [(4 * column + b_thread, b_byte) for column in range(N)]
    return rows, columns


def probe_words():
    lanes = torch.arange(32, dtype=torch.int32, device="cuda")
    return sum((PROBE_BASE + 4 * lanes + byte) << (8 * byte) for byte in range(4)).to(torch.int32)


def unit_words():
    return torch.full((32,), 0x7F7F7F7F, dtype=torch.int32, device="cuda")


def probed_lanes(d, axis):
    exponents = torch.log2(d / K).round().long() + 127 - PROBE_BASE
    first = exponents[:, 0] if axis == 0 else exponents[0, :]
    return [(int(value) // 4, int(value) % 4) for value in first]


def test_f8f6f4_matches_the_fp64_product():
    cuda_seed(31)
    declare_gpu_work(1)
    a, b = random_e4m3(M, K), random_e4m3(N, K)
    expected, magnitude = product(a, b)
    check_close(run_mma(F8F6F4, a, b), expected, magnitude)


@pytest.mark.parametrize("selectors", list(BLOCK_SCALE))
def test_block_scale_selectors_pick_the_documented_lanes(selectors):
    declare_gpu_work(2)
    ones_a = torch.ones(M, K, device="cuda").to(torch.float8_e4m3fn)
    ones_b = torch.ones(N, K, device="cuda").to(torch.float8_e4m3fn)
    rows, columns = expected_lanes(selectors)
    by_row = run_mma(BLOCK_SCALE[selectors], ones_a, ones_b, probe_words(), unit_words())
    by_column = run_mma(BLOCK_SCALE[selectors], ones_a, ones_b, unit_words(), probe_words())
    assert probed_lanes(by_row, 0) == rows, f"rows of A take (lane, byte) {probed_lanes(by_row, 0)}"
    assert probed_lanes(by_column, 1) == columns, f"columns of B take (lane, byte) {probed_lanes(by_column, 1)}"


@pytest.mark.parametrize("selectors", list(BLOCK_SCALE))
def test_block_scale_matches_the_scaled_fp64_product(selectors):
    cuda_seed(32)
    declare_gpu_work(1)
    a, b = random_e4m3(M, K), random_e4m3(N, K)
    scale_a_bytes = torch.randint(120, 135, (32, 4), dtype=torch.uint8, device="cuda")
    scale_b_bytes = torch.randint(120, 135, (32, 4), dtype=torch.uint8, device="cuda")
    rows, columns = expected_lanes(selectors)
    scale_a = torch.stack([scale_a_bytes[lane, byte] for lane, byte in rows])
    scale_b = torch.stack([scale_b_bytes[lane, byte] for lane, byte in columns])
    expected, magnitude = product(a, b, scale_a, scale_b)
    result = run_mma(BLOCK_SCALE[selectors], a, b, scale_a_bytes.view(torch.int32).flatten(),
                     scale_b_bytes.view(torch.int32).flatten())
    check_close(result, expected, magnitude)


def swizzled_128(tile):
    """Returns the bytes that a 128-byte swizzle copy writes: chunk `c` of row `r` lands at chunk `c ^ (r % 8)`."""
    rows = torch.arange(tile.shape[0], device=tile.device)[:, None, None]
    chunks = torch.arange(TILE_COLUMNS // 16, device=tile.device)[None, :, None]
    offsets = torch.arange(16, device=tile.device)[None, None, :]
    out = torch.empty_like(tile)
    out[rows, (chunks ^ (rows % 8)) * 16 + offsets] = tile[rows, chunks * 16 + offsets]
    return out


def test_tma_load_swizzles_and_store_restores_tiles():
    function = kernel(TMA_COPY)
    declare_gpu_work(1)
    row_tiles, column_tiles = 3, 2
    source = torch.randint(0, 256, (row_tiles * TILE_ROWS, column_tiles * TILE_COLUMNS), dtype=torch.uint8,
                           device="cuda")
    target = torch.zeros_like(source)
    raw = torch.zeros(row_tiles * column_tiles, TILE_ROWS, TILE_COLUMNS, dtype=torch.uint8, device="cuda")
    box = (TILE_ROWS, TILE_COLUMNS)
    maps = tensormap.tensor_map(source, box), tensormap.tensor_map(target, box)
    launch(function, "mmpuu", (*maps, raw.data_ptr(), row_tiles, column_tiles), threads=128)
    for tile in range(row_tiles * column_tiles):
        row, column = tile // column_tiles * TILE_ROWS, tile % column_tiles * TILE_COLUMNS
        expected = swizzled_128(source[row:row + TILE_ROWS, column:column + TILE_COLUMNS].contiguous())
        assert torch.equal(raw[tile], expected), f"tile {tile} has another shared memory layout"
    assert torch.equal(target, source)
