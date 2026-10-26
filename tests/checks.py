import contextlib

import torch

queued_checks = []
case_labels = []

OUTPUT_RTOL = 2
BIT_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@contextlib.contextmanager
def case(**parameters):
    case_labels.append(" ".join(f"{name}={value}" for name, value in parameters.items()))
    try:
        yield
    finally:
        case_labels.pop()


def queue(kind, name, value):
    queued_checks.append((kind, " ".join([*case_labels, name]), value))


def rounding_step(tensor_ref, dtype):
    reference = tensor_ref.float()
    return 2 * (reference.to(dtype).float().abs() * torch.finfo(dtype).eps).amax()


def queue_tolerance_check(tensor, tensor_ref, tensor_pt, name, dtype=None):
    """Queues the check that the largest error of `tensor` against `tensor_ref` is at most `OUTPUT_RTOL` times that
    of the PyTorch result `tensor_pt`, plus a rounding step of the output type."""
    assert tensor.shape == tensor_ref.shape, f"{name}: shape {tuple(tensor.shape)} != {tuple(tensor_ref.shape)}"
    atol = rounding_step(tensor_ref, dtype or tensor.dtype)
    error = (tensor.float() - tensor_ref.float()).abs().amax()
    allowed = OUTPUT_RTOL * (tensor_pt.float() - tensor_ref.float()).abs().amax() + atol
    queue("bound", name, torch.stack([error, allowed]))


def same_bits(tensor, expected):
    bits = BIT_DTYPES[tensor.element_size()]
    return (tensor.view(bits) == expected.view(bits)) | (tensor.float().isnan() & expected.float().isnan())


def queue_equal_check(tensor, expected, name):
    """Queues the check that `tensor` and `expected` hold the same bits, except that any NaN matches any NaN."""
    assert (tensor.shape, tensor.dtype) == (expected.shape, expected.dtype), \
        f"{name}: {tuple(tensor.shape)} {tensor.dtype} != {tuple(expected.shape)} {expected.dtype}"
    queue("failure", name, (~same_bits(tensor, expected)).any().reshape(1))


def queue_requirement(name, condition):
    queue("failure", name, ~condition.reshape(1))


def verify_queued_checks():
    checks = list(queued_checks)
    queued_checks.clear()
    if not checks:
        return
    values = torch.cat([value.float().reshape(-1) for _, _, value in checks]).tolist()
    failures = []
    offset = 0
    for kind, name, value in checks:
        part = values[offset : offset + value.numel()]
        offset += value.numel()
        if kind == "bound" and not part[0] <= part[1]:
            failures.append(f"{name}: error {part[0]:.3e} exceeds {part[1]:.3e}")
        elif kind == "failure" and part[0]:
            failures.append(f"{name} failed")
    assert not failures, f"{len(failures)} of {len(checks)} checks failed:\n" + "\n".join(failures)
