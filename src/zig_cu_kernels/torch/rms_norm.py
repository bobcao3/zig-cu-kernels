import functools
import inspect
import math
from collections import namedtuple
from typing import Optional

import torch

from .. import rms_norm as launcher
from ..rms_norm import Options
from ..tensors import dtype_name
from .common import (
    ELEMENTS,
    activation_block_scales,
    current_stream_handle,
    leading_rows,
    require_choice,
    require_one_element,
    require_scale_dtype,
    require_tensorwise_scale,
    writes_in_place,
)
from .ops import Output, define

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "rms_norm_fp8_quant",
    "fused_add_rms_norm_fp8_quant",
    "adaptive_rms_norm",
    "fused_add_adaptive_rms_norm",
    "fused_add_adaptive_rms_norm_fp8_quant",
    "dual_rms_norm",
    "dual_rms_norm_fp8_quant",
    "rms_norm_op",
    "rms_norm_out_op",
    "fused_add_rms_norm_op",
    "fused_add_rms_norm_out_op",
    "rms_norm_fp8_quant_op",
    "rms_norm_fp8_quant_out_op",
    "fused_add_rms_norm_fp8_quant_op",
    "fused_add_rms_norm_fp8_quant_out_op",
    "adaptive_rms_norm_op",
    "adaptive_rms_norm_out_op",
    "fused_add_adaptive_rms_norm_op",
    "fused_add_adaptive_rms_norm_out_op",
    "fused_add_adaptive_rms_norm_fp8_quant_op",
    "fused_add_adaptive_rms_norm_fp8_quant_out_op",
    "dual_rms_norm_op",
    "dual_rms_norm_out_op",
    "dual_rms_norm_fp8_quant_op",
    "dual_rms_norm_fp8_quant_out_op",
]

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
WEIGHT_DTYPES = (torch.bfloat16, torch.float32)
COMPUTED_SCALINGS = ("rowwise", "block128")
TENSORS = ("x", "residual", "branch", "branch_weight", "weight", "gate", "scale", "shift", "fp8_scale")
MODULATION = ("gate", "scale", "shift")
STATIC_TYPES = {"eps": float, "zero_centered": bool, "rows_per_sample": Optional[int], "fp8_scaling": str,
                "fp8_scale_dtype": torch.dtype, "out_dtype": torch.dtype}
DTYPE_PARAMETERS = ("out_dtype", "fp8_scale_dtype")
ARGUMENT_DEFAULTS = {"eps": 0.0, "zero_centered": False, "rows_per_sample": None, "fp8_scaling": None,
                     "fp8_scale_dtype": None, "out_dtype": None}


class Arguments(namedtuple("Arguments", (*TENSORS, *ARGUMENT_DEFAULTS),
                           defaults=(None,) * (len(TENSORS) - 1) + tuple(ARGUMENT_DEFAULTS.values()))):
    """The arguments of one call of an RMS norm op. A field that the op does not take keeps its default."""

    __slots__ = ()

    @property
    def rows(self):
        return math.prod(self.x.shape[:-1])

    def inputs(self):
        return {name: getattr(self, name) for name in TENSORS if getattr(self, name) is not None}


def require_dtype_of(name, dtype, allowed):
    if dtype is not None and dtype not in allowed:
        raise ValueError(f"{name} must be a {' or '.join(map(str, allowed))} tensor, got {dtype}")


def output_of(fp8_scale, fp8_scaling, fp8_scale_dtype, out_dtype):
    """Checks the output arguments and returns the `output` option of the kernel."""
    if fp8_scaling is not None:
        require_choice("fp8_scaling", fp8_scaling, COMPUTED_SCALINGS)
        require_scale_dtype("fp8_scale_dtype", fp8_scale_dtype, fp8_scaling)
        if fp8_scaling == "rowwise":
            return "fp8_rowwise"
        return "fp8_blockwise_e8m0" if fp8_scale_dtype == E8M0 else "fp8_blockwise"
    if fp8_scale is not None:
        require_dtype_of("fp8_scale", fp8_scale, (torch.float32,))
        if out_dtype not in (None, FP8):
            raise ValueError(f"with fp8_scale the output dtype must be {FP8}, got {out_dtype}")
        return "fp8_tensorwise"
    if out_dtype is None:
        return "bf16"
    require_choice("out_dtype", out_dtype, tuple(ELEMENTS))
    return ELEMENTS[out_dtype]


def options_of(a):
    dtypes = tuple(None if tensor is None else tensor.dtype for tensor in a[: len(TENSORS)])
    return options_for(dtypes, a.x.dim() > 0, a.zero_centered, a.fp8_scaling, a.fp8_scale_dtype, a.out_dtype)


@functools.lru_cache(maxsize=4096)
def options_for(dtypes, has_rows, zero_centered, fp8_scaling, fp8_scale_dtype, out_dtype):
    x, residual, branch, branch_weight, weight, gate, scale, shift, fp8_scale = dtypes
    require_dtype_of("x", x, tuple(ELEMENTS))
    if not has_rows:
        raise ValueError("x needs at least one dimension")
    require_dtype_of("residual", residual, (x,))
    for name, dtype in (("branch", branch), ("gate", gate), ("scale", scale), ("shift", shift)):
        require_dtype_of(name, dtype, (BF16,))
    require_dtype_of("weight", weight, WEIGHT_DTYPES)
    require_dtype_of("branch_weight", branch_weight, (weight,))
    if weight is None and zero_centered:
        raise ValueError("zero_centered needs a weight")
    weight_option = "none" if weight is None else ("zero_centered_" if zero_centered else "") + ELEMENTS[weight]
    if residual is not None:
        add = "residual"
    elif branch_weight is not None:
        add = "dual" if gate is None else "dual_gated"
    else:
        add = "none" if branch is None else "gated"
    modulation = "none" if scale is None else "scale" if shift is None else "scale_shift"
    output = output_of(fp8_scale, fp8_scaling, fp8_scale_dtype, out_dtype)
    return Options(x=ELEMENTS[x], add=add, weight=weight_option, modulation=modulation, output=output)


def check(a):
    options = options_of(a)
    launcher.require_built(options)
    return options


def require_shape(name, tensor, shape):
    if tensor is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have the shape {tuple(shape)}, got {tuple(tensor.shape)}")


def rows_per_sample_of(a):
    if a.rows_per_sample is not None:
        return a.rows_per_sample
    if a.rows == 0:
        return 1
    samples = next(math.prod(getattr(a, name).shape[:-1]) for name in MODULATION if getattr(a, name) is not None)
    if samples == 0 or a.rows % samples:
        raise ValueError(f"{a.rows} rows do not split into {samples} samples; pass rows_per_sample")
    return a.rows // samples


def check_sizes(a):
    options = options_of(a)
    width = a.x.shape[-1]
    for name in ("residual", "branch"):
        require_shape(name, getattr(a, name), a.x.shape)
    for name in ("weight", "branch_weight"):
        require_shape(name, getattr(a, name), (width,))
    if width % launcher.VALUES_PER_CHUNK:
        raise ValueError(f"the row length must be a multiple of {launcher.VALUES_PER_CHUNK}, got {width}")
    if options.blockwise and width % launcher.SCALE_GROUP:
        raise ValueError(f"fp8_scaling='block128' needs a row length that is a multiple of {launcher.SCALE_GROUP}")
    launcher.size_for(options, width)
    if a.rows > launcher.MAX_U32:
        raise ValueError(f"x has {a.rows} rows, more than 2^32 - 1")
    if a.fp8_scale is not None:
        require_one_element("fp8_scale", a.fp8_scale)
    if not options.modulated:
        return
    if a.rows_per_sample is not None and not 1 <= a.rows_per_sample <= launcher.MAX_U32:
        raise ValueError(f"rows_per_sample must lie between 1 and 2^32 - 1, got {a.rows_per_sample}")
    samples = -(-a.rows // rows_per_sample_of(a))
    for name in MODULATION:
        tensor = getattr(a, name)
        if tensor is not None and (tensor.dim() < 1 or tensor.shape[-1] != width or
                                   math.prod(tensor.shape[:-1]) < samples):
            raise ValueError(f"{name} must have at least {samples} rows of {width} values, got {tuple(tensor.shape)}")


def y_like(a):
    if a.out_dtype is not None:
        return a.x.new_empty(a.x.shape, dtype=a.out_dtype)
    quantized = a.fp8_scale is not None or a.fp8_scaling is not None
    return a.x.new_empty(a.x.shape, dtype=FP8 if quantized else BF16)


def y_dtype(a):
    return BF16 if a.fp8_scale is None else FP8


def scales_like(a):
    if a.fp8_scaling == "rowwise":
        return a.x.new_empty((a.rows, 1), dtype=torch.float32)
    return activation_block_scales(a.x, a.fp8_scale_dtype)


def residual_out_like(a):
    return a.x.new_empty(a.x.shape)


def ready(name, tensor, stream):
    """Returns `tensor`, or a contiguous copy when the kernels cannot read `tensor` in place."""
    dtype = dtype_name(tensor.dtype)
    if name in ("weight", "branch_weight"):
        fits = launcher.fits(launcher.vector_of, name, tensor, dtype, tensor.shape[-1], stream)
    elif name == "fp8_scale":
        fits = tensor.data_ptr() % tensor.element_size() == 0
    else:
        fits = launcher.fits(launcher.matrix_of, name, tensor, dtype, stream)
    return tensor if fits else tensor.clone(memory_format=torch.contiguous_format)


def contiguous_like(tensor):
    return torch.empty_like(tensor, memory_format=torch.contiguous_format)


def blockwise_scales_like(scales):
    return scales.new_empty((scales.shape[1], scales.shape[0])).t()


def target(output, layout_fits, inputs, aliases=(), temporary=contiguous_like):
    if layout_fits and writes_in_place(output, inputs, aliases):
        return output
    return temporary(output)


def launch(options, a, out, scales_out, residual_out, stream, inputs=None):
    """Plans and launches the kernel; `inputs` replaces the tensors of `a` that the kernel reads."""
    inputs = a.inputs() if inputs is None else inputs
    rows_per_sample = rows_per_sample_of(a) if options.modulated else 1
    planned = launcher.plan(options, inputs["x"], out, residual=inputs.get("residual"), branch=inputs.get("branch"),
                            residual_out=residual_out, weight=inputs.get("weight"),
                            branch_weight=inputs.get("branch_weight"), gate=inputs.get("gate"),
                            scale=inputs.get("scale"), shift=inputs.get("shift"),
                            fp8_scales=inputs.get("fp8_scale", scales_out), rows_per_sample=rows_per_sample,
                            stream=stream)
    if planned is not None:
        planned_launch, pointers = planned
        planned_launch.launch(pointers, float(a.eps), stream)


def row_stride(tensor, row_bytes=launcher.CHUNK_BYTES, output=False):
    """Returns the row stride in elements that the kernels take for `tensor`, or None when `tensor` needs a copy."""
    rows = leading_rows(tensor)
    width = tensor.shape[-1]
    if rows is None or (width > 1 and tensor.stride(-1) != 1) or tensor.data_ptr() % launcher.CHUNK_BYTES:
        return None
    count, stride = rows
    if count <= 1:
        return 0
    if count > launcher.MAX_U32:
        return None
    if stride * tensor.element_size() % row_bytes or stride > launcher.MAX_U32 or (output and stride < width):
        return None
    return stride


def vector_ready(tensor):
    return (tensor.shape[-1] <= 1 or tensor.stride(-1) == 1) and tensor.data_ptr() % launcher.CHUNK_BYTES == 0


def scales_stride(options, scales, rows):
    """Returns the stride that the kernels take for the FP8 scales `scales`, or None when `scales` needs a temporary."""
    if scales.data_ptr() % scales.element_size():
        return None
    if options.output == "fp8_rowwise":
        stride = scales.stride(0) if rows > 1 else 1
        return stride if 1 <= stride <= launcher.MAX_U32 else None
    scale_groups = scales.shape[1]
    if rows > 1 and scale_groups and scales.stride(0) != 1:
        return None
    stride = scales.stride(1) if scale_groups > 1 else rows
    return stride if rows <= stride <= launcher.MAX_U32 else None


def launch_in_place(options, a, out, scales_out, residual_out, device_index, stream):
    """Launches the kernel for the call `a` if the kernel can read every input and write every output in place, and
    returns True after a launch."""
    rows, width = a.rows, a.x.shape[-1]
    if rows == 0 or width == 0 or (options.add != "none") != (residual_out is not None):
        return False
    if (options.fp8 and options.output != "fp8_tensorwise") != (scales_out is not None):
        return False
    other = a.residual if options.add == "residual" else a.branch
    row_inputs = [tensor for tensor in (a.x, other) if tensor is not None]
    weights = [tensor for tensor in (a.weight, a.branch_weight) if tensor is not None]
    modulation = [getattr(a, name) for name in MODULATION]
    others = [*weights, *(tensor for tensor in modulation if tensor is not None)]
    row_input_strides = [row_stride(tensor) for tensor in row_inputs]
    modulation_strides = [0 if tensor is None else row_stride(tensor) for tensor in modulation]
    if None in row_input_strides or None in modulation_strides or not all(map(vector_ready, weights)):
        return False
    if a.fp8_scale is not None:
        if a.fp8_scale.data_ptr() % a.fp8_scale.element_size():
            return False
        others.append(a.fp8_scale)
    y_stride = row_stride(out, launcher.FP8_CHUNK_BYTES if options.fp8 else launcher.CHUNK_BYTES, output=True)
    out_aliases = [tensor for tensor in row_inputs if tensor.dtype == out.dtype]
    out_inputs = others + [tensor for tensor in row_inputs if tensor.dtype != out.dtype]
    if y_stride is None or not writes_in_place(out, out_inputs, out_aliases):
        return False
    residual_out_stride = 0
    if residual_out is not None:
        residual_out_stride = row_stride(residual_out, output=True)
        if residual_out_stride is None or not writes_in_place(residual_out, others, row_inputs):
            return False
    fp8_stride = 0
    if scales_out is not None:
        fp8_stride = scales_stride(options, scales_out, rows)
        if fp8_stride is None or not writes_in_place(scales_out, [*others, *row_inputs]):
            return False
    fp8 = a.fp8_scale if a.fp8_scale is not None else scales_out
    strides = (*(row_input_strides + [0])[:2], residual_out_stride, *modulation_strides, y_stride, fp8_stride)
    sizes = (rows, width, rows_per_sample_of(a) if options.modulated else 1)
    pointers = tuple(0 if tensor is None else tensor.data_ptr()
                     for tensor in (a.x, other, residual_out, a.weight, a.branch_weight, *modulation, out, fp8))
    launcher.launch_for(device_index, options, strides, sizes).launch(pointers, float(a.eps), stream)
    return True


def run(a, out, scales_out=None, residual_out=None):
    """Writes the results of the call `a` into the outputs, through copies and temporaries for the tensors that the
    launcher rejects with ValueError."""
    options = options_of(a)
    rows, width = a.rows, a.x.shape[-1]
    require_shape("out", out, a.x.shape)
    require_shape("residual_out", residual_out, a.x.shape)
    if scales_out is not None:
        require_shape("scales_out", scales_out, (rows, 1) if options.output == "fp8_rowwise" else
                      (rows, width // launcher.SCALE_GROUP))
    device_index = a.x.get_device()
    stream = current_stream_handle(device_index)
    if launch_in_place(options, a, out, scales_out, residual_out, device_index, stream):
        return
    try:
        return launch(options, a, out, scales_out, residual_out, stream)
    except ValueError:
        pass
    inputs = {name: ready(name, tensor, stream) for name, tensor in a.inputs().items()}
    row_input_names = [name for name in ("x", "residual", "branch") if name in inputs]
    others = [tensor for name, tensor in inputs.items() if name not in row_input_names]
    row_inputs = [inputs[name] for name in row_input_names]
    out_dtype = launcher.OUTPUT_DTYPES[options.output]
    row_bytes = launcher.FP8_CHUNK_BYTES if options.fp8 else launcher.CHUNK_BYTES
    out_fits = launcher.fits(launcher.matrix_of, "out", out, out_dtype, stream, row_bytes, output=True)
    out_aliases = [tensor for tensor in row_inputs if tensor.dtype == out.dtype]
    out_target = target(out, out_fits, others + [tensor for tensor in row_inputs if tensor.dtype != out.dtype],
                        out_aliases)
    residual_out_target = None
    if residual_out is not None:
        x_dtype = launcher.X_DTYPES[options.x]
        residual_out_fits = launcher.fits(launcher.matrix_of, "residual_out", residual_out, x_dtype, stream,
                                          output=True)
        residual_out_target = target(residual_out, residual_out_fits, others, row_inputs)
    scales_target = None
    if scales_out is not None:
        scales_fits = launcher.fits(launcher.fp8_scales_of, options, scales_out, rows, width, stream)
        temporary = blockwise_scales_like if options.blockwise else contiguous_like
        scales_target = target(scales_out, scales_fits, [*others, *row_inputs], temporary=temporary)
    launch(options, a, out_target, scales_target, residual_out_target, stream, inputs)
    for output, written in ((out, out_target), (residual_out, residual_out_target), (scales_out, scales_target)):
        if written is not output:
            output.copy_(written)


def parameters_of(text):
    """Returns the parameters of a signature such as `x, weight?, eps`, where `?` marks an optional tensor."""
    parameters = []
    for item in text.split(", "):
        name = item.rstrip("?")
        if name in TENSORS:
            annotation = Optional[torch.Tensor] if item.endswith("?") else torch.Tensor
        else:
            annotation = STATIC_TYPES[name]
        parameters.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation))
    return parameters


def taking(parameters, body, returns=inspect.Signature.empty):
    """Returns a function with the signature `parameters` that calls `body` with the arguments as a dict."""
    names = [parameter.name for parameter in parameters]

    def function(*values):
        return body(dict(zip(names, values)))

    function.__signature__ = inspect.Signature(parameters, return_annotation=returns)
    return function


def define_norm(name, signature, outputs):
    """Registers the ops `name` and `<name>_out`, where `outputs` maps each output to an allocator and a dtype rule."""
    functional = parameters_of(signature)
    inputs = [parameter for parameter in functional if parameter.name not in DTYPE_PARAMETERS]
    written = [inspect.Parameter(output, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=torch.Tensor)
               for output in outputs]
    input_names = [parameter.name for parameter in inputs]
    has_out_dtype = "out_dtype" in signature

    def run_values(values):
        a = Arguments(**{key: values[key] for key in input_names}, out_dtype=values["out"].dtype if has_out_dtype else
                      None, fp8_scale_dtype=values["scales_out"].dtype if "scales_out" in values else None)
        run(a, **{output: values[output] for output in outputs})

    def call(function):
        return taking(functional, lambda values: function(Arguments(**values)))

    allocations = {output: Output(taking(functional, lambda values, allocate=allocate: allocate(Arguments(**values)),
                                         torch.Tensor), dtype if not callable(dtype) else call(dtype))
                   for output, (allocate, dtype) in outputs.items()}
    return define(name, run=taking(inputs + written, run_values, None), outputs=allocations, check=call(check),
                  check_sizes=call(check_sizes))


Y = (y_like, "out_dtype")
RESIDUAL_OUT = (residual_out_like, "x")
SCALES = (scales_like, "fp8_scale_dtype")
QUANTIZED = (y_like, FP8)
NORMED = (y_like, BF16)

RMS_NORM = define_norm("rms_norm", "x, weight?, fp8_scale?, eps, zero_centered, out_dtype", {"out": Y})
FUSED_ADD = define_norm("fused_add_rms_norm", "x, residual, weight?, fp8_scale?, eps, zero_centered, out_dtype",
                        {"out": Y, "residual_out": RESIDUAL_OUT})
RMS_NORM_FP8 = define_norm("rms_norm_fp8_quant", "x, weight?, eps, zero_centered, fp8_scaling, fp8_scale_dtype",
                           {"out": QUANTIZED, "scales_out": SCALES})
FUSED_ADD_FP8 = define_norm(
    "fused_add_rms_norm_fp8_quant", "x, residual, weight?, eps, zero_centered, fp8_scaling, fp8_scale_dtype",
    {"out": QUANTIZED, "scales_out": SCALES, "residual_out": RESIDUAL_OUT})
ADAPTIVE = define_norm("adaptive_rms_norm", "x, scale, shift?, weight?, eps, zero_centered, rows_per_sample?",
                       {"out": NORMED})
FUSED_ADD_ADAPTIVE = define_norm(
    "fused_add_adaptive_rms_norm", "x, branch, gate, scale, shift?, weight?, eps, zero_centered, rows_per_sample?",
    {"out": NORMED, "residual_out": RESIDUAL_OUT})
FUSED_ADD_ADAPTIVE_FP8 = define_norm(
    "fused_add_adaptive_rms_norm_fp8_quant",
    "x, branch, gate, scale, shift?, weight?, eps, zero_centered, rows_per_sample?, fp8_scaling, fp8_scale_dtype",
    {"out": QUANTIZED, "scales_out": SCALES, "residual_out": RESIDUAL_OUT})
DUAL = define_norm(
    "dual_rms_norm",
    "x, branch, branch_weight, weight, gate?, scale?, fp8_scale?, eps, zero_centered, rows_per_sample?",
    {"out": (y_like, y_dtype), "residual_out": RESIDUAL_OUT})
DUAL_FP8 = define_norm(
    "dual_rms_norm_fp8_quant",
    "x, branch, branch_weight, weight, gate?, scale?, eps, zero_centered, rows_per_sample?, fp8_scaling, "
    "fp8_scale_dtype",
    {"out": QUANTIZED, "scales_out": SCALES, "residual_out": RESIDUAL_OUT})

rms_norm_op, rms_norm_out_op = RMS_NORM.functional, RMS_NORM.out
fused_add_rms_norm_op, fused_add_rms_norm_out_op = FUSED_ADD.functional, FUSED_ADD.out
rms_norm_fp8_quant_op, rms_norm_fp8_quant_out_op = RMS_NORM_FP8.functional, RMS_NORM_FP8.out
fused_add_rms_norm_fp8_quant_op, fused_add_rms_norm_fp8_quant_out_op = FUSED_ADD_FP8.functional, FUSED_ADD_FP8.out
adaptive_rms_norm_op, adaptive_rms_norm_out_op = ADAPTIVE.functional, ADAPTIVE.out
fused_add_adaptive_rms_norm_op = FUSED_ADD_ADAPTIVE.functional
fused_add_adaptive_rms_norm_out_op = FUSED_ADD_ADAPTIVE.out
fused_add_adaptive_rms_norm_fp8_quant_op = FUSED_ADD_ADAPTIVE_FP8.functional
fused_add_adaptive_rms_norm_fp8_quant_out_op = FUSED_ADD_ADAPTIVE_FP8.out
dual_rms_norm_op, dual_rms_norm_out_op = DUAL.functional, DUAL.out
dual_rms_norm_fp8_quant_op, dual_rms_norm_fp8_quant_out_op = DUAL_FP8.functional, DUAL_FP8.out


def given_scale(fp8_scaling, fp8_scale, fp8_scale_dtype, scales_out):
    """Checks the scale arguments of a `_fp8_quant` function and returns whether the call takes the caller's scale."""
    if fp8_scaling != "tensorwise":
        if fp8_scale is not None:
            raise ValueError(f"fp8_scaling={fp8_scaling!r} computes the scales and takes no fp8_scale")
        return False
    if fp8_scale is None:
        raise ValueError("fp8_scaling='tensorwise' needs fp8_scale")
    require_tensorwise_scale("fp8_scale", fp8_scale)
    require_scale_dtype("fp8_scale_dtype", fp8_scale_dtype, fp8_scaling)
    if scales_out is not None:
        raise ValueError("fp8_scaling='tensorwise' returns fp8_scale and takes no scales_out")
    return True


def resolved_dtype(x, out_dtype):
    return getattr(x, "dtype", None) if out_dtype is None else out_dtype


def rms_norm(x, weight, eps, *, zero_centered=False, out_dtype=None, out=None):
    """Returns the RMS norm of the rows of `x`, times `weight` unless it is None (README section "RMS norm")."""
    return RMS_NORM(x, weight, None, eps, zero_centered, resolved_dtype(x, out_dtype), outputs={"out": out})


def fused_add_rms_norm(x, residual, weight, eps, *, zero_centered=False, out_dtype=None, out=None,
                       residual_out=None):
    """Returns `(y, residual_out)`, the RMS norm of the residual stream `x + residual` and the residual stream.
    See the README section "RMS norm"."""
    return FUSED_ADD(x, residual, weight, None, eps, zero_centered, resolved_dtype(x, out_dtype),
                     outputs={"out": out, "residual_out": residual_out})


def rms_norm_fp8_quant(x, weight, eps, *, fp8_scaling, fp8_scale=None, fp8_scale_dtype=torch.float32,
                       zero_centered=False, out=None, scales_out=None):
    """Returns the E4M3 codes `q` of the RMS norm of `x` and their `scales` for the scaling type `fp8_scaling`.
    See the README section "RMS norm"."""
    if given_scale(fp8_scaling, fp8_scale, fp8_scale_dtype, scales_out):
        return RMS_NORM(x, weight, fp8_scale, eps, zero_centered, FP8, outputs={"out": out}), fp8_scale.reshape(())
    return RMS_NORM_FP8(x, weight, eps, zero_centered, fp8_scaling, fp8_scale_dtype,
                        outputs={"out": out, "scales_out": scales_out})


def fused_add_rms_norm_fp8_quant(x, residual, weight, eps, *, fp8_scaling, fp8_scale=None,
                                 fp8_scale_dtype=torch.float32, zero_centered=False, out=None, scales_out=None,
                                 residual_out=None):
    """Returns `(q, scales, residual_out)`, the results of `fused_add_rms_norm` quantized as in `rms_norm_fp8_quant`."""
    if given_scale(fp8_scaling, fp8_scale, fp8_scale_dtype, scales_out):
        q, h = FUSED_ADD(x, residual, weight, fp8_scale, eps, zero_centered, FP8,
                         outputs={"out": out, "residual_out": residual_out})
        return q, fp8_scale.reshape(()), h
    return FUSED_ADD_FP8(x, residual, weight, eps, zero_centered, fp8_scaling, fp8_scale_dtype,
                         outputs={"out": out, "scales_out": scales_out, "residual_out": residual_out})


def adaptive_rms_norm(x, scale, shift=None, *, eps, weight=None, zero_centered=False, rows_per_sample=None, out=None):
    """Returns the adaptive RMS norm `(1 + scale) * n + shift` of the rows of `x`. See the README section "RMS norm"."""
    return ADAPTIVE(x, scale, shift, weight, eps, zero_centered, rows_per_sample, outputs={"out": out})


def fused_add_adaptive_rms_norm(x, branch, gate, scale, shift=None, *, eps, weight=None, zero_centered=False,
                                rows_per_sample=None, out=None, residual_out=None):
    """Returns `(y, residual_out)`, the adaptive RMS norm of the residual stream `bf16(x + bf16(gate * branch))` and
    the residual stream. See the README section "RMS norm"."""
    return FUSED_ADD_ADAPTIVE(x, branch, gate, scale, shift, weight, eps, zero_centered, rows_per_sample,
                              outputs={"out": out, "residual_out": residual_out})


def fused_add_adaptive_rms_norm_fp8_quant(x, branch, gate, scale, shift=None, *, eps, fp8_scaling,
                                          fp8_scale_dtype=torch.float32, weight=None, zero_centered=False,
                                          rows_per_sample=None, out=None, scales_out=None, residual_out=None):
    """Returns `(q, scales, residual_out)`, the results of `fused_add_adaptive_rms_norm` with computed FP8 scales."""
    return FUSED_ADD_ADAPTIVE_FP8(x, branch, gate, scale, shift, weight, eps, zero_centered, rows_per_sample,
                                  fp8_scaling, fp8_scale_dtype,
                                  outputs={"out": out, "scales_out": scales_out, "residual_out": residual_out})


def dual_rms_norm(x, branch, branch_weight, weight, eps, *, zero_centered=False, gate=None, scale=None,
                  rows_per_sample=None, out=None, residual_out=None):
    """Returns `(y, residual_out)`, where `residual_out` is `x + gate * n` for the RMS norm `n` of `branch`, or `x + n`
    without `gate`, and `y` is the normalized `residual_out`. See the README section "RMS norm"."""
    return DUAL(x, branch, branch_weight, weight, gate, scale, None, eps, zero_centered, rows_per_sample,
                outputs={"out": out, "residual_out": residual_out})


def dual_rms_norm_fp8_quant(x, branch, branch_weight, weight, eps, *, fp8_scaling, fp8_scale=None,
                            fp8_scale_dtype=torch.float32, zero_centered=False, gate=None, scale=None,
                            rows_per_sample=None, out=None, scales_out=None, residual_out=None):
    """Returns `(q, scales, residual_out)`, the results of `dual_rms_norm` quantized as in `rms_norm_fp8_quant`."""
    if given_scale(fp8_scaling, fp8_scale, fp8_scale_dtype, scales_out):
        q, h = DUAL(x, branch, branch_weight, weight, gate, scale, fp8_scale, eps, zero_centered, rows_per_sample,
                    outputs={"out": out, "residual_out": residual_out})
        return q, fp8_scale.reshape(()), h
    return DUAL_FP8(x, branch, branch_weight, weight, gate, scale, eps, zero_centered, rows_per_sample, fp8_scaling,
                    fp8_scale_dtype, outputs={"out": out, "scales_out": scales_out, "residual_out": residual_out})
