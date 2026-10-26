import pytest
import torch

import pack_references as ref
from conftest import cuda_seed
from opcheck import check_op
from zig_cu_kernels import fp8_pack, runtime
from zig_cu_kernels import torch as ops

KERNELS = ("mxfp8_quant_bf16", "fp8_b128_quant_f32_f32", "fp8_b128_quant_weight_bf16_e8m0", "fp8_rowwise_quant_f32",
           "fp8_rowwise_quant_held_f32", "mxfp8_quant_dual_bf16")
ACTIVATION_SHAPES = ((1, 128), (7, 640), (130, 640), (257, 5376), (128, 128), (300, 128), (65, 5376))
WEIGHT_SHAPES = ((128, 128), (256, 640), (384, 5376))
DTYPES = (torch.bfloat16, torch.float32)
EDGES = (448.0, -448.0, 464.0, 447.99, 480.0, 25.0, 17.0)
TIES = (448.0, *range(17, 32, 2))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not all(runtime.is_available(name) for name in KERNELS),
    reason="needs a GPU that runs the fp8_pack kernels",
)


def lattice(rows, cols, dtype, seed=0):
    """Returns normal values with row magnitudes from 1e-3 to 1e3, one column times 40, a zero first row, a zero
    block of 128 values, saturation-edge values times powers of two and E4M3 ties."""
    cuda_seed(seed)
    x = torch.randn(rows, cols, device="cuda") * torch.logspace(-3, 3, rows, device="cuda")[:, None]
    x[:, cols // 3] *= 40
    x[0] = 0
    if rows > 1:
        powers = torch.arange(cols, device="cuda") % 21 - 10
        edges = torch.tensor(EDGES, device="cuda").repeat(cols // len(EDGES) + 1)[:cols]
        x[1] = edges * torch.exp2(powers.float())
    if rows > 2 and cols >= 256:
        x[2, 128:256] = 0
    if rows > 3:
        x[3] = torch.tensor(TIES, device="cuda").repeat(cols // len(TIES) + 1)[:cols]
    return x.to(dtype)


def assert_same_bits(actual, expected):
    for name, a, e in zip(("values", "scales"), actual, expected):
        assert a.shape == e.shape and a.dtype == e.dtype, (name, a.shape, a.dtype, e.shape, e.dtype)
        a_bytes, e_bytes = a.contiguous().view(torch.uint8), e.contiguous().view(torch.uint8)
        differing = (a_bytes != e_bytes).sum().item()
        assert differing == 0, f"{differing} of {e_bytes.numel()} bytes of the {name} differ"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
def test_mxfp8_quant(shape, dtype):
    x = lattice(*shape, dtype)
    assert_same_bits(fp8_pack.mxfp8_quant(x), ref.mxfp8_quant(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_mxfp8_quant_3d(dtype):
    x = torch.stack([lattice(130, 640, dtype, seed) for seed in range(3)])
    assert_same_bits(fp8_pack.mxfp8_quant(x), ref.mxfp8_quant(x))


def test_mxfp8_quant_pads_columns():
    x = lattice(33, 160, torch.bfloat16)
    assert_same_bits(fp8_pack.mxfp8_quant(x), ref.mxfp8_quant(x))


@pytest.mark.parametrize("e8m0", (True, False))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
def test_fp8_b128_quant(shape, dtype, e8m0):
    x = lattice(*shape, dtype)
    assert_same_bits(fp8_pack.fp8_b128_quant(x, e8m0=e8m0), ref.fp8_b128_quant(x, e8m0=e8m0))


@pytest.mark.parametrize("e8m0", (True, False))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", WEIGHT_SHAPES)
def test_fp8_b128_quant_weight(shape, dtype, e8m0):
    w = lattice(*shape, dtype)
    assert_same_bits(fp8_pack.fp8_b128_quant_weight(w, e8m0=e8m0), ref.fp8_b128_quant_weight(w, e8m0=e8m0))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", ACTIVATION_SHAPES)
def test_fp8_rowwise_quant(shape, dtype):
    x = lattice(*shape, dtype)
    assert_same_bits(fp8_pack.fp8_rowwise_quant(x), ref.fp8_rowwise_quant(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fp8_rowwise_quant_3d(dtype):
    x = torch.stack([lattice(65, 640, dtype, seed) for seed in range(2)])
    assert_same_bits(fp8_pack.fp8_rowwise_quant(x), ref.fp8_rowwise_quant(x))


def test_every_bf16_value():
    bits = torch.arange(1 << 16, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    finite = bits[bits.float().isfinite()]
    x = torch.nn.functional.pad(finite, (0, -finite.numel() % 5376)).view(-1, 5376)
    assert_same_bits(fp8_pack.mxfp8_quant(x), ref.mxfp8_quant(x))
    assert_same_bits(fp8_pack.fp8_b128_quant(x, e8m0=True), ref.fp8_b128_quant(x, e8m0=True))
    assert_same_bits(fp8_pack.fp8_b128_quant(x, e8m0=False), ref.fp8_b128_quant(x, e8m0=False))
    assert_same_bits(fp8_pack.fp8_rowwise_quant(x), ref.fp8_rowwise_quant(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_subnormal_and_huge_magnitudes(dtype):
    cuda_seed(1)
    exponents = torch.linspace(-148, 124, 256, device="cuda").round()
    x = (torch.randn(256, 640, device="cuda") * torch.exp2(exponents)[:, None]).to(dtype)
    assert_same_bits(fp8_pack.mxfp8_quant(x), ref.mxfp8_quant(x))
    for e8m0 in (True, False):
        assert_same_bits(fp8_pack.fp8_b128_quant(x, e8m0=e8m0), ref.fp8_b128_quant(x, e8m0=e8m0))
        assert_same_bits(fp8_pack.fp8_b128_quant_weight(x, e8m0=e8m0), ref.fp8_b128_quant_weight(x, e8m0=e8m0))
    assert_same_bits(fp8_pack.fp8_rowwise_quant(x), ref.fp8_rowwise_quant(x))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", (*ACTIVATION_SHAPES, (33, 160), (512, 384)))
def test_mxfp8_quant_columnwise_and_dual(shape, dtype):
    x = lattice(*shape, dtype)
    columnwise = ref.mxfp8_quant_columnwise(x)
    assert_same_bits(fp8_pack.mxfp8_quant_columnwise(x), columnwise)
    dual = fp8_pack.mxfp8_quant_dual(x)
    assert_same_bits(dual[:2], ref.mxfp8_quant(x))
    assert_same_bits(dual[2:], columnwise)


def test_mxfp8_quant_columnwise_of_transpose():
    """The columnwise pack of the transposed lattice puts the edge and tie rows of the lattice into columns."""
    x = lattice(640, 130, torch.float32).t().contiguous()[:, :128].contiguous()
    assert_same_bits(fp8_pack.mxfp8_quant_columnwise(x), ref.mxfp8_quant_columnwise(x))


@pytest.mark.parametrize("dtype", DTYPES)
def test_mxfp8_quant_into_caller_buffers(dtype):
    x = lattice(130, 640, dtype)
    values = torch.empty(130, 640, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty(2, 5, 32, 16, dtype=torch.uint8, device="cuda")
    fp8_pack.mxfp8_quant_(x, values, scales)
    assert_same_bits((values, scales), ref.mxfp8_quant(x))


OP_CASES = {
    "mxfp8_quant": (lambda x: (x,), ref.mxfp8_quant),
    "mxfp8_quant_columnwise": (lambda x: (x,), ref.mxfp8_quant_columnwise),
    "mxfp8_quant_dual": (lambda x: (x,), ref.mxfp8_quant_dual),
    "fp8_b128_quant": (lambda x: (x, True), lambda x: ref.fp8_b128_quant(x, e8m0=True)),
    "fp8_b128_quant_weight": (lambda x: (x[:256], False), lambda x: ref.fp8_b128_quant_weight(x[:256], e8m0=False)),
    "fp8_rowwise_quant": (lambda x: (x,), ref.fp8_rowwise_quant),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", OP_CASES)
def test_quantizer_ops_match_the_references_and_pass_opcheck(name, dtype):
    arguments_of, reference = OP_CASES[name]
    x = lattice(257, 640, dtype)
    op = getattr(ops, name)
    result = op(*arguments_of(x))
    expected = reference(x)
    for start in range(0, len(expected), 2):
        assert_same_bits(result[start:start + 2], expected[start:start + 2])
    check_op(op, *arguments_of(x))


def test_mxfp8_quant_in_place_op():
    x = lattice(257, 640, torch.float32)
    values = torch.empty(257, 640, dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty(3, 5, 32, 16, dtype=torch.uint8, device="cuda")
    ops.mxfp8_quant_(x, values, scales)
    assert_same_bits((values, scales), ref.mxfp8_quant(x))
    check_op(ops.mxfp8_quant_, x, values, scales)
    with pytest.raises(ValueError, match="scales must be"):
        ops.mxfp8_quant_(x, values, scales[:2])
    unaligned = torch.empty(257 * 640 + 8, dtype=torch.float8_e4m3fn, device="cuda")[8:].view(257, 640)
    with pytest.raises(ValueError, match="values must start"):
        ops.mxfp8_quant_(x, unaligned, scales)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", (16, 4096 + 16, 16384, 16384 + 16))
def test_fp8_rowwise_quant_at_the_held_row_limit(width, dtype):
    """Rows up to 16384 values stay in registers; wider rows take the kernel that reads each row twice."""
    x = lattice(9, width, dtype)
    assert_same_bits(fp8_pack.fp8_rowwise_quant(x), ref.fp8_rowwise_quant(x))
