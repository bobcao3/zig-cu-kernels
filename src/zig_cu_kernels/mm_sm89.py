"""The `mm_abt` kernels of compute capability 8.9: the `fp8_gemm_sm89` kernels, which read the MXFP8 scale atoms and
the block128 activation scale atoms directly. The MXFP8 and block128 `mm_glu` without `Z` run one fused kernel when the
composed chain of `glu_quant` would make the same launch, so that both give the same bits, and the composed chain
otherwise. The rowwise `mm_glu` without `Z` writes BF16 `h` from the fused kernel and then quantizes its rows. With
`Z`, the composed chain measured faster on an RTX 6000 Ada. Products whose row count lies in `SHARED_TILE_ROWS` may let
work groups share the k tiles of a tile, which measured faster on an RTX 6000 Ada at its power limit."""

import contextlib

import torch

from . import fp8_gemm, fp8_mm, glu_quant, packs, runtime

ARCH = 89
E8M0 = torch.float8_e8m0fnu
SHARED_TILE_ROWS = range(65, 257)


def scale_view(scales):
    return scales.view(E8M0) if scales.dtype == fp8_mm.E8M0_BYTE else scales


def shares_tiles(m):
    return m in SHARED_TILE_ROWS


@contextlib.contextmanager
def tile_workspace(problem):
    """Yields the `workspace` argument of the `fp8_gemm` launchers for `problem`."""
    if not shares_tiles(problem.m):
        yield None
        return
    from .torch import common

    with common.workspace_for(problem.device_index, problem.stream) as workspace:
        yield workspace.addresses


def fuses(family, m, columns, k, tilings):
    """Whether the `mm_abt` product of the composed chain runs one of `tilings` without shared tiles, so that the fused
    GLU kernel gives the same bits."""
    choice = fp8_gemm.store_choice(family, m, columns, k, shares_tiles(m))
    return choice.tiling in tilings and choice == fp8_gemm.store_choice(family, m, columns, k)


def store(problem, scale_a, scale_b):
    tensors = problem.epilogue.tensors
    assert problem.epilogue.kind in ("store", "bias"), f"{problem.epilogue.kind} epilogues need another kernel"
    with tile_workspace(problem) as workspace:
        fp8_gemm.gemm(problem.a.values, problem.b.values, scale_view(scale_a), scale_view(scale_b),
                      problem.outputs["y"], bias=tensors.get("bias"), workspace=workspace, stream=problem.stream)


def fp8_b128_mm_abt(problem):
    tensors = problem.epilogue.tensors
    assert problem.epilogue.kind in ("store", "bias"), f"{problem.epilogue.kind} epilogues need another kernel"
    with tile_workspace(problem) as workspace:
        fp8_gemm.fp8_b128_gemm(problem.a.values, problem.b.values, scale_view(problem.a.scales),
                               scale_view(problem.b.scales[:, :problem.k // packs.GROUP]), problem.outputs["y"],
                               bias=tensors.get("bias"), workspace=workspace, stream=problem.stream)


def mxfp8_mm_abt(problem):
    tensors = problem.epilogue.tensors
    m, n = problem.m, problem.n
    low, up = tensors.get("low"), tensors.get("up")
    with tile_workspace(problem) as workspace:
        fp8_gemm.mxfp8_gemm(problem.a.values[:m], problem.b.values[:n], scale_view(problem.a.scales),
                            scale_view(problem.b.scales), problem.outputs["y"], bias=tensors.get("bias"),
                            residual=tensors.get("residual"), low=low, up=None if up is None else up[:n],
                            workspace=workspace, stream=problem.stream)


def fp8_b128_mm_glu(problem):
    parameters, outputs = problem.epilogue.parameters, problem.outputs
    hidden, k = parameters["hidden"], problem.k
    z = outputs["z"]
    if z is not None or not fuses("block128", problem.m, problem.n, k, fp8_gemm.GLU_TILINGS):
        glu_quant.mm_glu(problem)
        return
    values, scales = problem.b.values, scale_view(problem.b.scales[:, :k // packs.GROUP])
    halves = [(values[:hidden], scales[:hidden // packs.GROUP]),
              (values[hidden:2 * hidden], scales[hidden // packs.GROUP:2 * hidden // packs.GROUP])]
    gate = 0 if parameters["gate_first"] else 1
    s_h = scale_view(outputs["s_h"])
    if hidden // packs.GROUP % packs.ATOM_BLOCKS:
        with torch.cuda.stream(torch.cuda.ExternalStream(problem.stream)):
            outputs["s_h"].zero_()
    fp8_gemm.fp8_b128_glu_gemm(problem.a.values[:problem.m], halves[gate][0], halves[1 - gate][0],
                               scale_view(problem.a.scales), halves[gate][1], halves[1 - gate][1], outputs["h"], s_h,
                               activation=parameters["activation"], stream=problem.stream)


def mxfp8_mm_glu(problem):
    parameters, outputs = problem.epilogue.parameters, problem.outputs
    padded, k = problem.n // 2, problem.k
    if outputs["z"] is not None or not fuses("mxfp8", problem.m, problem.n, k, fp8_gemm.FAMILY_GLU_TILINGS["mxfp8"]):
        glu_quant.mm_glu(problem)
        return
    values, scales = problem.b.values, scale_view(problem.b.scales)
    atoms = padded // packs.GROUP
    halves = [(values[:padded], scales[:atoms]), (values[padded:2 * padded], scales[atoms:2 * atoms])]
    gate = 0 if parameters["gate_first"] else 1
    fp8_gemm.mxfp8_glu_gemm(problem.a.values[:problem.m], halves[gate][0], halves[1 - gate][0],
                            scale_view(problem.a.scales), halves[gate][1], halves[1 - gate][1], outputs["h"],
                            scale_view(outputs["s_h"]), activation=parameters["activation"],
                            hidden=parameters["hidden"], clamp_limit=glu_quant.bf16_limit(parameters["clamp_limit"]),
                            stream=problem.stream)


def fp8_rowwise_mm_abt(problem):
    store(problem, problem.a.scales, problem.b.scales)


def fp8_rowwise_mm_glu(problem):
    parameters, outputs = problem.epilogue.parameters, problem.outputs
    hidden, m = parameters["hidden"], problem.m
    if outputs["z"] is not None or not fuses("rowwise", m, problem.n, problem.k, fp8_gemm.GLU_TILINGS):
        glu_quant.mm_glu(problem)
        return
    values, scales = problem.b.values, problem.b.scales
    halves = [(values[:hidden], scales[:hidden]), (values[hidden:2 * hidden], scales[hidden:2 * hidden])]
    gate = 0 if parameters["gate_first"] else 1
    h_bf16 = problem.workspace(m * hidden * 2).view(torch.bfloat16).view(m, hidden)
    fp8_gemm.fp8_rowwise_glu_gemm(problem.a.values[:m], halves[gate][0], halves[1 - gate][0], problem.a.scales[:m],
                                  halves[gate][1], halves[1 - gate][1], h_bf16, activation=parameters["activation"],
                                  stream=problem.stream)
    quant = "fp8_rowwise_quant_bf16"
    rows_per_group = runtime.info(quant).threads // 32
    glu_quant.run(quant, (packs.ceil_div(m, rows_per_group), 1, 1), glu_quant.QUANT_SIGNATURE,
                  (h_bf16.data_ptr(), outputs["h"].data_ptr(), outputs["s_h"].data_ptr(), m, hidden), problem.stream)


fp8_mm.register(ARCH, "mxfp8", "mm_abt", mxfp8_mm_abt)
fp8_mm.register(ARCH, "fp8_b128_e8m0", "mm_abt", fp8_b128_mm_abt)
fp8_mm.register(ARCH, "fp8_b128_fp32", "mm_abt", fp8_b128_mm_abt)
fp8_mm.register(ARCH, "fp8_rowwise", "mm_abt", fp8_rowwise_mm_abt)
fp8_mm.register(ARCH, "mxfp8", "mm_glu", mxfp8_mm_glu)
fp8_mm.register(ARCH, "fp8_b128_e8m0", "mm_glu", fp8_b128_mm_glu)
fp8_mm.register(ARCH, "fp8_b128_fp32", "mm_glu", fp8_b128_mm_glu)
fp8_mm.register(ARCH, "fp8_rowwise", "mm_glu", fp8_rowwise_mm_glu)
