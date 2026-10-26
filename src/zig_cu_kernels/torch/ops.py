import inspect
import types
import typing
from typing import Callable, NamedTuple, Optional

import torch
from torch._library.autograd import make_autograd_impl

from .. import runtime
from .common import (
    device_index_of,
    fp32_scalar,
    require_int,
    require_same_device,
    require_separate_outputs,
    require_tensor,
)

NAMESPACE = "zig_cu_kernels"
CHECKED_TYPES = (torch.Tensor, float, int)
E8M0 = torch.float8_e8m0fnu
LIBRARIES = {}
AFTER_AUTOGRAD = torch._C._after_autograd_keyset
WITHOUT_FORMULA = types.SimpleNamespace(_backward_fn=None, _setup_context_fn=None)


class Output(NamedTuple):
    """`dtype` is a `torch.dtype`, the name of a required tensor input or `torch.dtype` parameter of the functional op,
    or a function of the arguments of the functional op. Such a function reads only dtypes, static arguments and which
    optional tensors are None. `keyword` names the output in the `outputs` of a public call."""

    allocate: Callable
    dtype: object
    keyword: Optional[str] = None


def annotations_of(function):
    parameters = inspect.signature(function, eval_str=True).parameters
    return {name: parameter.annotation for name, parameter in parameters.items()}


def base_type(annotation):
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        options = [option for option in typing.get_args(annotation) if option is not type(None)]
        return (options[0], True) if len(options) == 1 else (None, True)
    return annotation, False


def tensor_positions(annotations):
    return [index for index, annotation in enumerate(annotations.values()) if base_type(annotation)[0] is torch.Tensor]


def holds_e8m0(values):
    return any(value.dtype == E8M0 if isinstance(value, torch.Tensor) else value is E8M0 for value in values)


def library_of(namespace):
    if namespace not in LIBRARIES:
        LIBRARIES[namespace] = torch.library.Library(namespace, "FRAGMENT")
    return LIBRARIES[namespace]


def writing_below_autograd(operator, output_count):
    def kernel(keyset, *arguments):
        torch._C._increment_version(arguments[-output_count:])
        with torch._C._AutoDispatchBelowAutograd():
            return operator.redispatch(keyset & AFTER_AUTOGRAD, *arguments)

    return kernel


class OpPair:
    def __init__(self, name, run, outputs, check, check_sizes, namespace):
        self.name, self.run, self.check, self.check_sizes = name, run, check, check_sizes
        self.outputs = list(outputs.values())
        self.output_names = list(outputs)
        self.keywords = [output.keyword or name for name, output in outputs.items()]
        functional, out = annotations_of(self.outputs[0].allocate), annotations_of(run)
        self.functional_names, out_names = list(functional), list(out)
        if any(list(annotations_of(output.allocate)) != self.functional_names for output in self.outputs):
            raise ValueError(f"the allocation functions of {name} must take the same parameters")
        self.input_count = len(out_names) - len(self.output_names)
        input_names = out_names[: self.input_count]
        if out_names[self.input_count :] != self.output_names or not set(input_names) <= set(functional):
            raise ValueError(f"{run.__name__} must take inputs of the functional op {name}, then {self.output_names}")
        self.input_positions = [self.functional_names.index(input_name) for input_name in input_names]
        statics = self.read_dtype_rules(functional, input_names)
        if set(self.functional_names) - set(input_names) != set(statics):
            raise ValueError(f"every parameter of the functional op {name} must be an input of {run.__name__} or the "
                             "dtype of an output")
        self.functional_recipe = [(True, input_names.index(parameter)) if parameter in input_names
                                  else (False, statics[parameter]) for parameter in self.functional_names]
        self.functional_tensors, self.out_tensors = tensor_positions(functional), tensor_positions(out)
        if not self.functional_tensors or base_type(functional[self.functional_names[self.functional_tensors[0]]])[1]:
            raise ValueError(f"the first tensor parameter of {name} must be a tensor that is never None")
        self.functional_tensor_names = [self.functional_names[index] for index in self.functional_tensors]
        self.out_tensor_names = [out_names[index] for index in self.out_tensors]
        self.functional_devices = list(zip(self.functional_tensor_names, self.functional_tensors))
        self.out_devices = list(zip(self.out_tensor_names, self.out_tensors))
        self.keyword_set = frozenset(self.keywords)
        self.checked_parameters = []
        for index, (parameter, annotation) in enumerate(functional.items()):
            kind, optional = base_type(annotation)
            if kind in CHECKED_TYPES:
                self.checked_parameters.append((index, parameter, kind, optional))
        self.register(namespace)

    def read_dtype_rules(self, functional, input_names):
        """Returns the dtype parameters of the functional op, each with the index of the first output that takes it."""
        self.output_rules, self.dtype_parameters, statics = [], [], {}
        for index, (name, output) in enumerate(zip(self.output_names, self.outputs)):
            if isinstance(output.dtype, torch.dtype):
                self.output_rules.append(("fixed", output.dtype))
            elif isinstance(output.dtype, str) and output.dtype in input_names and \
                    base_type(functional[output.dtype]) == (torch.Tensor, False):
                self.output_rules.append(("input", input_names.index(output.dtype)))
            elif isinstance(output.dtype, str) and functional.get(output.dtype) is torch.dtype:
                self.output_rules.append(("static", statics.setdefault(output.dtype, index)))
                self.dtype_parameters.append((index, self.functional_names.index(output.dtype)))
            elif callable(output.dtype):
                self.output_rules.append(("computed", output.dtype))
            else:
                raise ValueError(f"the dtype of {name} in {self.name} is no dtype, no function and no name of a "
                                 "required tensor input or a dtype parameter")
        return statics

    def register(self, namespace):
        returns = "Tensor" if len(self.outputs) == 1 else f"({', '.join(['Tensor'] * len(self.outputs))})"
        parameters = torch.library.infer_schema(self.outputs[0].allocate, mutates_args=()).rsplit(" -> ", 1)[0]
        out_schema = torch.library.infer_schema(self.run, mutates_args=self.output_names)
        library = library_of(namespace)
        out_name = f"{self.name}_out"
        for name, schema, kernel, fake in ((self.name, f"{parameters} -> {returns}", self.functional_kernel,
                                            self.functional_fake),
                                           (out_name, out_schema, self.out_kernel, self.out_fake)):
            library.define(name + schema, tags=(torch.Tag.pt2_compliant_tag,))
            library.impl(name, kernel, "CUDA")
            torch.library.register_fake(f"{namespace}::{name}", fake, lib=library)
        operators = getattr(torch.ops, namespace)
        self.functional, self.out = getattr(operators, self.name).default, getattr(operators, out_name).default
        library.impl(self.name, make_autograd_impl(self.functional, WITHOUT_FORMULA), "Autograd", with_keyset=True)
        library.impl(out_name, writing_below_autograd(self.out, len(self.outputs)), "Autograd", with_keyset=True)

    def allocated(self, arguments):
        if len(self.outputs) == 1:
            return self.outputs[0].allocate(*arguments)
        return tuple(output.allocate(*arguments) for output in self.outputs)

    def inputs(self, arguments):
        return [arguments[index] for index in self.input_positions]

    def functional_arguments(self, arguments):
        outputs = arguments[self.input_count :]
        functional = [arguments[index] if is_input else outputs[index].dtype
                      for is_input, index in self.functional_recipe]
        for name, output, (kind, value) in zip(self.output_names, outputs, self.output_rules):
            if kind == "fixed":
                expected = value
            elif kind == "input":
                expected = arguments[value].dtype
            elif kind == "static":
                expected = outputs[value].dtype
            else:
                expected = value(*functional)
            if output.dtype != expected:
                raise ValueError(f"{name} must be a {expected} tensor, got {output.dtype}")
        return functional

    @torch._disable_dynamo
    def functional_kernel(self, *arguments):
        device_index = device_index_of(arguments, self.functional_devices)
        self.check(*arguments)
        if self.check_sizes is not None:
            self.check_sizes(*arguments)
        runtime.bind_device(device_index)
        outputs = self.allocated(arguments)
        self.run(*self.inputs(arguments), *(outputs if isinstance(outputs, tuple) else (outputs,)))
        return outputs

    def functional_fake(self, *arguments):
        require_same_device(self.functional_tensor_names, [arguments[i] for i in self.functional_tensors])
        self.check(*arguments)
        return self.allocated(arguments)

    @torch._disable_dynamo
    def out_kernel(self, *arguments):
        device_index = device_index_of(arguments, self.out_devices)
        functional_arguments = self.functional_arguments(arguments)
        self.check(*functional_arguments)
        if self.check_sizes is not None:
            self.check_sizes(*functional_arguments)
        require_separate_outputs(self.output_names, arguments[self.input_count :])
        runtime.bind_device(device_index)
        self.run(*arguments)

    def out_fake(self, *arguments):
        require_same_device(self.out_tensor_names, [arguments[i] for i in self.out_tensors])
        self.check(*self.functional_arguments(arguments))

    def checked_arguments(self, arguments):
        if len(arguments) != len(self.functional_names):
            raise TypeError(f"{self.name} takes {len(self.functional_names)} arguments, got {len(arguments)}")
        checked = list(arguments)
        for index, name, kind, optional in self.checked_parameters:
            value = checked[index]
            if value is None and optional:
                continue
            if kind is torch.Tensor:
                if not isinstance(value, torch.Tensor):
                    require_tensor(name, value)
            else:
                checked[index] = fp32_scalar(name, value) if kind is float else require_int(name, value)
        return checked

    def given_outputs(self, outputs):
        """Returns the tensors of `outputs` in the order of the outputs and the number of outputs without a tensor."""
        if not outputs:
            return None, len(self.keywords)
        if not outputs.keys() <= self.keyword_set:
            unknown = next(keyword for keyword in outputs if keyword not in self.keyword_set)
            raise TypeError(f"{self.name} has no output {unknown}")
        given = [outputs.get(keyword) for keyword in self.keywords]
        missing = 0
        for keyword, output in zip(self.keywords, given):
            if output is None:
                missing += 1
            elif not isinstance(output, torch.Tensor):
                require_tensor(keyword, output)
        return given, missing

    def copied_results(self, arguments, given):
        """Computes new outputs with the functional op and copies them into the tensors of `given`."""
        results = self.functional(*arguments)
        results = (results,) if len(self.outputs) == 1 else results
        finals = []
        for keyword, output, result in zip(self.keywords, given, results):
            if output is not None:
                if output.dtype != result.dtype or output.shape != result.shape:
                    raise ValueError(f"{keyword} must be a {result.dtype} tensor of the shape {tuple(result.shape)}")
                output.copy_(result)
            finals.append(result if output is None else output)
        return finals[0] if len(finals) == 1 else tuple(finals)

    def __call__(self, *arguments, outputs=None):
        """Calls the `_out` op when `outputs` maps a keyword to a tensor, and the functional op otherwise. Under
        `torch.compile`, a call with an E8M0 tensor or dtype copies the results of the functional op instead, because
        Inductor of PyTorch 2.14.1 fails on a mutating custom op with a `float8_e8m0fnu` argument."""
        arguments = self.checked_arguments(arguments)
        given, missing = self.given_outputs(outputs)
        if missing == len(self.keywords):
            return self.functional(*arguments)
        for index, parameter in self.dtype_parameters:
            output, expected = given[index], arguments[parameter]
            if output is not None and output.dtype != expected:
                raise ValueError(f"{self.keywords[index]} must be a {expected} tensor, got {output.dtype}")
        if torch.compiler.is_compiling() and holds_e8m0((*arguments, *given)):
            return self.copied_results(arguments, given)
        if missing:
            if not torch.compiler.is_compiling():
                self.check(*arguments)
            given = [output if output is not None else spec.allocate(*arguments)
                     for spec, output in zip(self.outputs, given)]
        self.out(*self.inputs(arguments), *given)
        return given[0] if len(given) == 1 else tuple(given)


def define(name, *, run, outputs, check, check_sizes=None, namespace=NAMESPACE):
    """Registers the functional op `name` and the op `<name>_out` for CUDA tensors and returns them as an `OpPair`.
    `run(*inputs, *outputs)` writes the outputs that `outputs` maps to `Output` values. `check`, `check_sizes`, each
    `Output.allocate` and each computed dtype take the arguments of the functional op, and the `_out` op passes the
    dtype of each output for the dtype parameter of that output. The fake implementations skip `check_sizes`."""
    return OpPair(name, run, outputs, check, check_sizes, namespace)
