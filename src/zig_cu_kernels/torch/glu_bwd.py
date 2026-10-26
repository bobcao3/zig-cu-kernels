"""Custom op for the GLU backward. The fake implementation reads only shapes and dtypes."""

import torch
from torch._library.autograd import make_autograd_impl

from .. import glu_bwd as launcher
from .ops import NAMESPACE, WITHOUT_FORMULA, library_of

SCHEMAS = {
    "mxfp8_glu_backward": "(Tensor dh, Tensor z, bool gate_first=False, float? clamp_limit=None, "
                          "str activation=\"swiglu\") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
    "mxfp8_glu_backward_bf16": "(Tensor dh, Tensor z, bool gate_first=False, bool out_h=False, bool out_q=False, "
                               "float? clamp_limit=None, str activation=\"swiglu\") -> (Tensor, Tensor?, Tensor?, Tensor?)",
}


def mxfp8_glu_backward_fake(dh, z, gate_first=False, clamp_limit=None, activation="swiglu"):
    rows, hidden = launcher.check(dh, z, gate_first, clamp_limit, activation)
    return tuple(dh.new_empty(shape, dtype=dtype) for shape, dtype in launcher.output_shapes(rows, hidden))


def mxfp8_glu_backward_bf16_fake(dh, z, gate_first=False, out_h=False, out_q=False, clamp_limit=None,
                                 activation="swiglu"):
    rows, hidden = launcher.check(dh, z, gate_first, clamp_limit, activation)
    return tuple(None if shape is None else dh.new_empty(shape[0], dtype=shape[1])
                 for shape in launcher.bf16_output_shapes(rows, hidden, out_h, out_q))


def register():
    library = library_of(NAMESPACE)
    for name, schema in SCHEMAS.items():
        library.define(name + schema, tags=(torch.Tag.pt2_compliant_tag,))
        library.impl(name, getattr(launcher, name), "CUDA")
        torch.library.register_fake(f"{NAMESPACE}::{name}", globals()[f"{name}_fake"], lib=library)
        operator = getattr(getattr(torch.ops, NAMESPACE), name).default
        library.impl(name, make_autograd_impl(operator, WITHOUT_FORMULA), "Autograd", with_keyset=True)
        globals()[name] = globals()[f"{name}_op"] = operator


register()
__all__ = [*SCHEMAS, *(f"{name}_op" for name in SCHEMAS)]
