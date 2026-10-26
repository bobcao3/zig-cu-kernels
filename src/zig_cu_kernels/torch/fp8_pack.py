"""Custom ops for the pack quantizers. The checks of the fake implementations read only shapes and dtypes."""

import torch
from torch._library.autograd import make_autograd_impl

from .. import fp8_pack as launcher
from .ops import NAMESPACE, WITHOUT_FORMULA, library_of, writing_below_autograd

SCHEMAS = {
    "mxfp8_quant": "(Tensor x) -> (Tensor, Tensor)",
    "mxfp8_quant_columnwise": "(Tensor x) -> (Tensor, Tensor)",
    "mxfp8_quant_dual": "(Tensor x) -> (Tensor, Tensor, Tensor, Tensor)",
    "fp8_b128_quant": "(Tensor x, bool e8m0) -> (Tensor, Tensor)",
    "fp8_b128_quant_weight": "(Tensor w, bool e8m0) -> (Tensor, Tensor)",
    "fp8_rowwise_quant": "(Tensor x) -> (Tensor, Tensor)",
}
IN_PLACE_SCHEMAS = {
    "mxfp8_quant_": "(Tensor x, Tensor(a!) values, Tensor(b!) scales) -> ()",
}


def empty(x, shapes):
    return tuple(x.new_empty(shape, dtype=dtype) for shape, dtype in shapes)


def mxfp8_quant_fake(x):
    return empty(x, launcher.mxfp8_pack_shapes(x))


def mxfp8_quant_columnwise_fake(x):
    return empty(x, launcher.mxfp8_columnwise_shapes(x))


def mxfp8_quant_dual_fake(x):
    return mxfp8_quant_fake(x) + mxfp8_quant_columnwise_fake(x)


def fp8_b128_quant_fake(x, e8m0):
    return empty(x, launcher.fp8_b128_pack_shapes(x, e8m0))


def fp8_b128_quant_weight_fake(w, e8m0):
    return empty(w, launcher.fp8_b128_weight_pack_shapes(w, e8m0))


def fp8_rowwise_quant_fake(x):
    return empty(x, launcher.fp8_rowwise_pack_shapes(x))


def mxfp8_quant__fake(x, values, scales):
    launcher.require_outputs(x, ("values", "scales"), (values, scales), launcher.mxfp8_pack_shapes(x))


def fp8_b128_quant_cuda(x, e8m0):
    return launcher.fp8_b128_quant(x, e8m0=e8m0)


def fp8_b128_quant_weight_cuda(w, e8m0):
    return launcher.fp8_b128_quant_weight(w, e8m0=e8m0)


CUDA_KERNELS = {
    "mxfp8_quant": launcher.mxfp8_quant,
    "mxfp8_quant_columnwise": launcher.mxfp8_quant_columnwise,
    "mxfp8_quant_dual": launcher.mxfp8_quant_dual,
    "fp8_b128_quant": fp8_b128_quant_cuda,
    "fp8_b128_quant_weight": fp8_b128_quant_weight_cuda,
    "fp8_rowwise_quant": launcher.fp8_rowwise_quant,
    "mxfp8_quant_": launcher.mxfp8_quant_,
}


def register():
    library = library_of(NAMESPACE)
    operators = getattr(torch.ops, NAMESPACE)
    for name, schema in (SCHEMAS | IN_PLACE_SCHEMAS).items():
        library.define(name + schema, tags=(torch.Tag.pt2_compliant_tag,))
        library.impl(name, CUDA_KERNELS[name], "CUDA")
        torch.library.register_fake(f"{NAMESPACE}::{name}", globals()[f"{name}_fake"], lib=library)
        operator = getattr(operators, name).default
        if name in SCHEMAS:
            library.impl(name, make_autograd_impl(operator, WITHOUT_FORMULA), "Autograd", with_keyset=True)
        else:
            library.impl(name, writing_below_autograd(operator, 2), "Autograd", with_keyset=True)
        globals()[name] = globals()[f"{name}_op"] = operator


register()
__all__ = [*SCHEMAS, *IN_PLACE_SCHEMAS, *(f"{name}_op" for name in (*SCHEMAS, *IN_PLACE_SCHEMAS))]
