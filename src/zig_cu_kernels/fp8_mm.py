"""The launcher interface of the FP8 GEMMs over packs. A `Problem` describes one product with its epilogue, and a
table maps each GPU architecture and recipe to the kernels that compute the products."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from . import packs, runtime, tensormap

FP8 = torch.float8_e4m3fn
E8M0_BYTE = torch.uint8
BF16 = torch.bfloat16
ARCH_TARGETS = {89: "sm_89", 90: "sm_90a", 100: "sm_100f", 120: "sm_120f"}
RECIPES = ("mxfp8", "fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise")
OPERATIONS = ("mm_abt", "mm_atb", "mm_glu", "mm_qkv_prep")
EPILOGUES = ("store", "bias", "residual", "low_rank", "glu", "qkv_prep")
ACTIVATIONS = ("swiglu", "geglu_tanh")
LOW_RANKS = (32, 64, 128)
HEAD_DIMS = (64, 128, 256)
MAX_HEADS = 64
QKV_PLANE_COLUMNS = 256
WIDTH_MULTIPLE = 8
ROWWISE_K_MULTIPLE = 16


@dataclass(frozen=True)
class Pack:
    values: torch.Tensor
    scales: torch.Tensor


@dataclass
class Epilogue:
    kind: str
    tensors: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kind in EPILOGUES, f"epilogue {self.kind!r} is none of {EPILOGUES}"


@dataclass
class Problem:
    """One product `A Bᵀ` of the packs `a` `[m, k]` and `b` `[padded_n, k]` with the output width `n`, or `Aᵀ B` of the
    columnwise packs `[k, m]` and `[k, padded_n]` for the operation `mm_atb`. The kernel writes the tensors of
    `outputs`, encodes its tensor maps through `tensor_map` and takes a split-K workspace through `workspace`."""

    operation: str
    recipe: str
    a: Pack
    b: Pack
    m: int
    n: int
    padded_n: int
    k: int
    epilogue: Epilogue
    outputs: dict
    device_index: int
    stream: int
    tensor_maps: dict = field(default_factory=dict)
    workspaces: list = field(default_factory=list)

    def tensor_map(self, name, tensor, box, swizzle=128):
        if name not in self.tensor_maps:
            self.tensor_maps[name] = tensormap.tensor_map(tensor, box, swizzle=swizzle)
        return self.tensor_maps[name]

    def workspace(self, byte_count):
        buffer = torch.empty(byte_count, dtype=torch.uint8, device=torch.device("cuda", self.device_index))
        self.workspaces.append(buffer)
        return buffer


kernels: dict = {}


def register(arch: int, recipe: str, operation: str, kernel: Callable[[Problem], None]):
    assert arch in ARCH_TARGETS, f"arch {arch} is none of {tuple(ARCH_TARGETS)}"
    assert recipe in RECIPES, f"recipe {recipe!r} is none of {RECIPES}"
    assert operation in OPERATIONS, f"operation {operation!r} is none of {OPERATIONS}"
    kernels.setdefault((arch, recipe), {})[operation] = kernel


def kernel_name(operation, recipe, arch):
    return f"{recipe}_{operation}_sm{arch}"


def arch_of(capability: int) -> Optional[int]:
    major, minor = divmod(capability, 10)
    matches = [arch for arch, target in ARCH_TARGETS.items() if runtime.target_runs_on(target, major, minor)]
    return max(matches, default=None)


def lookup(capability: int, recipe: str, operation: str):
    arch = arch_of(capability)
    kernel = kernels.get((arch, recipe), {}).get(operation) if arch is not None else None
    if kernel is None:
        name = kernel_name(operation, recipe, arch if arch is not None else capability)
        major, minor = divmod(capability, 10)
        raise NotImplementedError(f"the kernel {name} does not exist for compute capability {major}.{minor}")
    return kernel


def available(recipe, operation, device_index=None):
    capability = runtime.compute_capability(torch.cuda.current_device() if device_index is None else device_index)
    try:
        lookup(capability, recipe, operation)
    except NotImplementedError:
        return False
    return True


def launch(problem: Problem):
    capability = runtime.compute_capability(problem.device_index)
    kernel = decode.lookup(capability, problem) if problem.operation == "mm_abt" else None
    (kernel or lookup(capability, problem.recipe, problem.operation))(problem)


def fp8_b128_recipe(scale_dtype):
    return "fp8_b128_e8m0" if scale_dtype == E8M0_BYTE else "fp8_b128_fp32"


def glu_shapes(recipe, m, padded_hidden, hidden, out_z, h_scale_dtype=E8M0_BYTE):
    z = ((m, 2 * hidden), BF16) if out_z else None
    if recipe == "mxfp8":
        return z, ((m, padded_hidden), FP8), (packs.mxfp8_scale_shape(m, padded_hidden), E8M0_BYTE)
    if recipe == "fp8_rowwise":
        return z, ((m, hidden), FP8), (packs.fp8_rowwise_scale_shape((m, hidden)), torch.float32)
    return z, ((m, hidden), FP8), (packs.fp8_b128_activation_scale_shape(m, hidden), h_scale_dtype)


def qkv_shapes(m, n_heads, head_dim, save_qkv):
    attn_dim = n_heads * head_dim
    return (m, attn_dim), (m, 2 * attn_dim), (m if save_qkv else 0, 3 * attn_dim)


from . import decode  # noqa: E402
from . import mm_sm89  # noqa: E402, F401
from . import mm_sm90  # noqa: E402, F401
from . import decode_sm90  # noqa: E402, F401
from . import mm_sm120  # noqa: E402, F401
from . import decode_sm120  # noqa: E402, F401
from . import mm_sm100  # noqa: E402, F401
from . import decode_sm100  # noqa: E402, F401
from . import glu_quant  # noqa: E402, F401
from . import qkv_prep  # noqa: E402, F401
