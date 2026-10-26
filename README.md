# zig-cu-kernels

Kernels for inference on NVIDIA GPUs of compute capability 8.x, 9.0, 10.x and 12.x, with PyTorch
custom ops:

- BF16 and FP8 GEMMs with fused epilogues. A GEMM is a general matrix multiplication. An epilogue
  is the bias, activation and residual add that a kernel applies to the product before the kernel
  stores the result.
- FP8 quantizers, which convert BF16 or FP32 tensors to FP8 values and their scales.
- RMS norms (root mean square normalizations).
- axpby, which computes `alpha * x + beta * y`.

FP8 values in this package are E4M3 codes (`torch.float8_e4m3fn`). The build compiles the kernels
ahead of time, from Zig through PTX to machine code. PTX is the virtual instruction set of NVIDIA
GPUs.

## Usage

```python
import torch
from zig_cu_kernels.torch import (fp8_gemm, fused_add_rms_norm, gated_gemm, gemm, quantize_fp8_rows,
                                  quantize_fp8_tiles)

x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
norm_weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
w_gate_up = torch.randn(2 * 11008, 4096, device="cuda", dtype=torch.bfloat16) / 64
w_down = torch.randn(4096, 11008, device="cuda", dtype=torch.bfloat16) / 105

h, residual = fused_add_rms_norm(x, residual, norm_weight, 1e-6)  # norm of x + residual, the sum
g = gated_gemm(h, w_gate_up[:11008], w_gate_up[11008:])  # SwiGLU: silu(h @ w_gate.T) * (h @ w_up.T)
y = gemm(g, w_down, residual=residual)  # g @ w_down.T + residual, shape (2, 512, 4096)

a, a_scales = quantize_fp8_rows(h)  # E4M3 codes, one scale per row and 128 values: (1024, 32)
b, b_scales = quantize_fp8_tiles(w_gate_up[:11008])  # one scale per 128 x 128 tile: (86, 32)
z = fp8_gemm(a, b, a_scales, b_scales, activation="silu")  # BF16, shape (2, 512, 11008)
```

## Custom ops

Importing `zig_cu_kernels.torch` registers the custom ops `torch.ops.zig_cu_kernels.<name>` and
`<name>_out`, and each function of the module calls one of these ops. Each op has a fake
implementation, which computes the shapes and dtypes of the results without running a kernel.
`zig_cu_kernels.torch` exports the two op objects as `<name>_op` and `<name>_out_op`. The functions
work inside CUDA graphs and under `torch.compile` with dynamic shapes. No op has a backward
formula, so a backward pass through a result raises `RuntimeError`.

Under `torch.compile`, a call that passes output tensors together with an E8M0 tensor or the E8M0
dtype (see [Scaling types](#scaling-types)) computes new results and copies them into the given
outputs. The functions avoid the `_out` op in this case because Inductor, the default compiler of
`torch.compile`, fails in PyTorch 2.14.1 on a mutating custom op with a `float8_e8m0fnu` argument.

## Outputs and aliasing

Each function takes its outputs as keyword arguments (`out`, and `residual_out`, `scales_out` or
`scale_out` where it has them) and returns the given tensors. An output that overlaps an input
gets its result through a temporary and a copy. These exact aliases need no temporary:

- `residual` may be `out` in the GEMMs.
- `out` and `residual_out` may be `x`, `residual` or `branch` in the RMS norms.
- `out` may be `x` or `y` in axpby.

An output whose elements share memory, and two outputs that share memory, raise `ValueError`.

## Layouts and strides

Most kernels read a tensor in place when its last dimension is contiguous, its leading dimensions
flatten into rows of one stride, and its data pointer and row stride are multiples of 16 bytes.
axpby needs contiguous tensors. An input that breaks these rules costs one copy, and an output
that breaks them gets its result through a temporary and a copy.

## Scalar arguments

`alpha`, `beta` and `eps` are Python numbers, which the kernels take rounded to FP32, to nearest
with ties to even. A tensor in their place raises `TypeError`, because reading the tensor would
wait for the device. A finite number that rounds to infinity in FP32 raises `OverflowError`. Under
`torch.compile`, each new value of one of these numbers compiles a new graph.

## Devices and streams

A call launches its kernels on the current PyTorch stream of the device of its tensors and returns
once the work is queued. A call never reads device memory on the host. The first call of a kernel
on a device loads the module of the kernel, and the load may wait for the device. A call may also
wait when the PyTorch caching allocator retries a failed `cudaMalloc`. Call each function once
with the shapes of a CUDA graph capture before the capture starts. The launchers take their stream
as an argument (see [Launchers](#launchers)).

## Launchers

The modules `zig_cu_kernels.gemm`, `fp8_gemm`, `rms_norm` and `axpby` hold launchers for callers
who manage their own buffers and streams. A launcher is a function that checks its arguments,
chooses a kernel and launches the kernel without a PyTorch op. The launchers run on the current
CUDA device of the calling thread and do not switch to the device of the tensors. They take
PyTorch tensors or objects that export DLPack or the CUDA Array Interface, and they raise
`ValueError` where a torch function would copy. A CPU tensor raises `ValueError`, and so does an
operand on a device other than the current device. The same holds for host memory, pinned or
pageable, that an object exports through the CUDA Array Interface. The error names the operand.

A launcher allocates no output, so the caller passes every output tensor. The docstring of each
launcher gives its arguments. These arguments differ from those of the torch functions:

- The BF16 GEMM launchers take no `out_dtype`, because the dtype of `out` selects the kernel.
- The launchers `fp8_gemm.quantize_rows` and `fp8_gemm.quantize_tiles` write the E4M3 codes to
  `codes` and the scales to `scales`. They take no `scale_dtype`, because the dtype of `scales`
  selects the kernel.
- The RMS norm launcher `rms_norm.rms_norm` takes an `Options` value that selects the kernel,
  `epsilon` for `eps`, `y` for `out` and one `fp8_scales` argument for the FP8 scales. Its
  `rows_per_sample` defaults to 1.
- The axpby launcher takes the dtype of its tensors as `element`, `"bf16"` or `"f32"`.

Each launcher queues its kernels on `stream`, which is a driver handle (an `int`), an object with
`__cuda_stream__`, or None for the default stream of the current device. A launcher passes `stream`
to the producer of each DLPack or CUDA Array Interface operand, so that the launch follows the
pending work of the producer. A launcher gives a PyTorch tensor no such ordering, so the caller
must make `stream` wait for the work that writes the tensor.

A product that splits k or uses stream-K needs memory for partial sums (see [Work groups, tiles and
splits](#work-groups-tiles-and-splits)). The BF16 launchers keep this memory in a workspace of the
device and stream, and during a CUDA graph capture they allocate and free the memory in the graph.
The FP8 launchers `fp8_gemm.gemm` and `fp8_gemm.gated_gemm` use stream-K only when the caller passes
`workspace`, a function that returns the device addresses of memory for the partial sums and their
counters. The docstring of `zig_cu_kernels.fp8_gemm.gemm` gives the arguments and the result of
`workspace`.

## Requirements

The package ships machine code and no PTX, so a GPU runs a kernel only when the build holds machine
code for the compute capability of the GPU. A build without `-Dsm` holds these targets:

| Kernels | Targets | Compute capability |
|---|---|---|
| BF16 GEMMs, axpby, RMS norms with a BF16 or FP32 result | sm_80, sm_86, sm_89 | 8.x |
| `fp8_gemm`, `fp8_gated_gemm`, the `quantize_fp8_` functions, RMS norms with an FP8 result | sm_89 | 8.9 |
| RMS norms with `ZIG_CU_KERNELS_RMS_NORM_ENGINE=bulk` (see [RMS norm](#rms-norm)) | sm_90, sm_100, sm_120 | 9.0, 10.x, 12.x |
| GEMM and decode kernels of the [FP8 operators](#fp8-operators) | sm_89, sm_90a, sm_100f, sm_120f | 8.9, 9.0, 10.x, 12.x |
| pack quantizers, the second kernels of the GLU and QKV paths, `mxfp8_glu_backward` and `mxfp8_glu_backward_bf16` | sm_89, sm_90, sm_100, sm_120 | 8.9, 9.0, 10.x, 12.x |

Machine code for sm_XY runs on compute capability X.Y and on the later minor versions of X. Code for
sm_90a runs on compute capability 9.0 only, and code for sm_100f or sm_120f runs on every device of
its family, 10.x or 12.x. On a GPU that runs none of the machine code of a kernel, a call raises
`RuntimeError` that names the compute capability of the GPU and the targets of the module of the
kernel. An FP8 operator without a kernel for the GPU raises `NotImplementedError` (see
[GPUs](#gpus)).

The kernels of the first two rows get machine code for 9.0, 10.x and 12.x only from a build that
names sm_90, sm_100 and sm_120. Without such a build, an RMS norm runs on those GPUs only through a
kernel of the third row, which `ZIG_CU_KERNELS_RMS_NORM_ENGINE=bulk` or `auto` selects for rows
that fit in its shared memory. `-Dsm` keeps only the targets that it names, so a build of every
target names all of them:

```sh
zig build -Dptxas="$CUDA_HOME/bin/ptxas" -Dsm=sm_80 -Dsm=sm_86 -Dsm=sm_89 -Dsm=sm_90 -Dsm=sm_90a \
    -Dsm=sm_100 -Dsm=sm_100f -Dsm=sm_120 -Dsm=sm_120f
```

`pip install` passes `ZIG_CU_KERNELS_SM` to the build as `-Dsm` and packages a build with `-Dsm`
only with `ZIG_CU_KERNELS_ALLOW_PARTIAL=1` (see [Build from source](#build-from-source)):

```sh
ZIG_CU_KERNELS_SM=sm_80,sm_86,sm_89,sm_90,sm_90a,sm_100,sm_100f,sm_120,sm_120f \
    ZIG_CU_KERNELS_ALLOW_PARTIAL=1 PTXAS="$CUDA_HOME/bin/ptxas" pip install .
```

The build needs a Zig compiler with the self-hosted PTX backend, and `ptxas` and `fatbinary` of
CUDA 13.0 or later.

The package checks no driver version. When the driver cannot load a cubin (the machine code file of
one module), the call raises `RuntimeError`, whose message names the kernel, the cubin, the error
of the driver and the driver version.

The package needs Python 3.11 or later, NumPy, cuda-python 13.4 or later and cuda-core 1.2 or later
with its `cu13` extra. The `torch` extra of the package requires torch 2.14.1 or a later 2.14
release. The launcher modules `gemm`, `fp8_gemm`, `rms_norm` and `axpby` import without torch.
`zig_cu_kernels.torch`, which holds the torch functions and the FP8 operators, imports torch, and
so do the other modules of the FP8 operators, such as `zig_cu_kernels.fp8_mm` and `fp8_pack`.

## BF16 GEMM

```python
gemm(a, b, *, bias=None, activation="none", residual=None, out=None, out_dtype=torch.bfloat16)
gated_gemm(a, w_gate, w_up, *, bias_gate=None, bias_up=None, activation="silu", residual=None,
           out=None)
```

`gemm` returns `F.linear(a, b, bias)` followed by an activation and a residual add. `a` is
`[..., k]`, `b` is `[n, k]`, `bias` is `[n]`, and `residual` has the output shape
`(*a.shape[:-1], n)`, where a row stride of 0 repeats one row. These tensors are BF16, and `bias`
stays BF16 for an FP32 output. `k` and `n` are multiples of 8, and `activation` is `"none"`,
`"silu"`, `"gelu_tanh"` or `"gelu"`. A BF16 output rounds to BF16 after the bias, after the
activation and after the residual, as the PyTorch BF16 CUDA ops do. With
`out_dtype=torch.float32`, `gemm` returns the FP32 sum plus the bias and takes no activation and no
residual.

`gated_gemm` computes a gated linear unit. The function returns `bf16(bf16(activation(g)) * v)`
plus a residual, with `g = bf16(a @ w_gate.T + bias_gate)` and `v = bf16(a @ w_up.T + bias_up)`.
`"silu"` gives SwiGLU, the two GELUs give GeGLU and `"none"` gives the bilinear unit. `w_gate` and
`w_up` are `[f, k]` and may be the two halves of one `[2f, k]` weight. `bias_gate` and `bias_up` are
`[f]`, and all operands are BF16.

## FP8 GEMM and quantizers

```python
fp8_gemm(a, b, scale_a, scale_b, *, bias=None, activation="none", residual=None,
         use_fast_accum=False, out=None)
fp8_gated_gemm(a, w_gate, w_up, scale_a, scale_gate, scale_up, *, bias_gate=None, bias_up=None,
               activation="silu", residual=None, use_fast_accum=False, out=None)
quantize_fp8_rows(x, *, scale_dtype=torch.float32, out=None, scales_out=None)  # (q, scales)
quantize_fp8_tiles(w, *, scale_dtype=torch.float32, out=None, scales_out=None)  # (q, scales)
quantize_fp8_tensorwise(x, scale=None, *, out=None, scale_out=None)  # (q, scale)
```

`a` (`[..., k]`, `m` rows) and `b` (`[n, k]`) are `torch.float8_e4m3fn`. The output is BF16 and
gets the epilogues of `gemm`, with BF16 biases and a BF16 residual.

### Scaling types

The scaling type says how the FP8 scales cover a tensor, and the shapes of the scales select it.
A scale group is a run of 128 consecutive values of k in one row of `a`, and the values of a scale
group share one block128 scale. E8M0 is the power-of-two scale dtype `torch.float8_e8m0fnu`.

| Scaling type | `scale_a` | `scale_b` | Scale dtype | `k` |
|---|---|---|---|---|
| tensorwise | one element | one element | FP32 | multiple of 16 |
| rowwise | `(m, 1)` | `(n, 1)` | FP32 | multiple of 16 |
| block128 | `(m, k / 128)` | `(ceil(n / 128), k / 128)` | FP32 or E8M0 | multiple of 128 |

`scale_a` may also keep the leading dimensions of `a`, and every scaling type needs `n % 8 == 0`
and `k > 0`. The weight scales follow the `[n, k]` shape of `b`, as in DeepGEMM, so a caller of
`torch._scaled_mm(a, w.t(), scale_a, s)` with rowwise or block128 scales passes `w` and `s.t()`.
The kernels read scale tensors at any strides whose leading dimensions flatten into one row stride.

### Accumulation

The FP8 tensor cores of compute capability 8.9 accumulate with fewer bits than FP32. By default, the
kernels therefore accumulate in chains. A chain is a run of at most 64 values of k that the `mma`
(matrix multiply-accumulate) instructions sum from zero before the kernel adds the chain to an FP32
sum. Block128 kernels also apply the scales of the chain in that addition. With
`use_fast_accum=True`, which tensorwise and rowwise scales take, the tensor cores accumulate all of
k that one warp (32 threads that execute together) computes, as
`torch._scaled_mm(..., use_fast_accum=True)` does.

### Quantizers

`quantize_fp8_rows` takes `x` of shape `[..., k]` with `k % 128 == 0` and returns one scale per
row and scale group. The scales have the shape `(m, k / 128)` and the strides `(1, m)` of the
BlockWise1x128 layout of `torch._scaled_mm`. `quantize_fp8_tiles` takes `w` of shape `[n, k]`
with `k % 128 == 0` and any `n` and returns one scale per 128 x 128 tile in a contiguous
`(ceil(n / 128), k / 128)` tensor.

Each scale is `amax / 448`, or 1 when amax is 0. Here amax is the largest magnitude among the
values that share the scale, and 448 is the largest finite E4M3 value. With
`scale_dtype=torch.float8_e8m0fnu`, the scale is the least power of two at or above that value. A
NaN among the values makes their scale NaN. Each code is the FP32 quotient of a value and its
scale, rounded to the nearest E4M3 value with ties to even. A quotient of magnitude above 448
becomes 448 with its sign.

`quantize_fp8_tensorwise` computes one FP32 scale for all of `x`, or quantizes with a given
one-element FP32 `scale` and returns the scale with shape `()`. The function needs `x` to have a
last dimension that is a multiple of 8, or a number of values that is a multiple of 8. The
quantizers take BF16 or FP32 inputs.

## Work groups, tiles and splits

A work group is a CUDA thread block. PTX calls it a CTA, short for cooperative thread array. The
threads of a work group run on one streaming multiprocessor (SM) and share one block of shared
memory. The GEMM kernels divide the output into rectangles called tiles, and a work group computes
a whole tile or a part of the k range of a tile. The raster order is the order in which the work
groups take the tiles. The order walks the tiles in panels of `panel` tile rows (order 0) or of
`panel` tile columns (order 1). The BF16 launchers take the raster order as `raster`, a pair
(panel, order). They also take `tiling`, the name of a built kernel. A tiled kernel has a name such
as `128x128x32`. A kernel for products of few rows has the name `m` followed by its largest row
count, such as `m8`. An unknown name raises `ValueError`, whose message lists the built names.

A product with few tiles can divide the k range of each tile among several work groups. In the BF16
kernels, the split of k gives each tile `splits` parts, and the last work group of a tile to finish
adds the FP32 partial sums of the parts in their order. The FP8 kernels use stream-K. The work
groups compute whole tiles first, one tile at a time, and then share the k ranges of the remaining
tiles evenly. The work groups that share a tile add their FP32 partial sums in a fixed order
through a workspace.

## RMS norm

| Computation | `y` | `(y, residual_out)` | `(q, scales)` | `(q, scales, residual_out)` |
|---|---|---|---|---|
| norm of `x`, or of `x + residual` | `rms_norm` | `fused_add_rms_norm` | `rms_norm_fp8_quant` | `fused_add_rms_norm_fp8_quant` |
| adaptive norm of `x`, or of `x + gate * branch` | `adaptive_rms_norm` | `fused_add_adaptive_rms_norm` | | `fused_add_adaptive_rms_norm_fp8_quant` |
| norm of `x` plus the norm of `branch` | | `dual_rms_norm` | | `dual_rms_norm_fp8_quant` |

`x` is `[..., d]`, where `d` is a multiple of 8 and at most 32768, or 16384 for an FP32 `x` in
`fused_add_rms_norm`. Each row `h` becomes `h * rsqrt(mean(h * h) + eps)`, computed in FP32, times
`weight`, or times `1 + weight` with `zero_centered=True`. A `weight` of None skips the
multiplication, and `zero_centered=True` without a weight raises `ValueError`. A BF16 weight
multiplies the normalized row rounded to the dtype of `x`, as `LlamaRMSNorm` of Hugging Face
transformers does, and zero-centered weights follow `GemmaRMSNorm`. The functions that start with
`fused_add_` also return the residual stream, `x + residual` in the dtype of `x` or
`bf16(x + bf16(gate * branch))`. `fused_add_rms_norm(x, r, w, eps, out=x, residual_out=r)` updates
both tensors in place, as the `fused_add_rms_norm` of vLLM does.

The adaptive functions apply the modulation `(1 + scale) * n + shift` in BF16 to the BF16 norm
`n`, as adaptive layer norm does. Each row of `gate`, `scale` and `shift` covers `rows_per_sample`
rows of `x`. By default, `rows_per_sample` is the row count of `x` divided by the row count of the
first of `gate`, `scale` and `shift` that the call passes. `dual_rms_norm` normalizes `branch` with
`branch_weight`, multiplies the result by `gate` when the call passes `gate`, adds the product to
`x` and normalizes the sum. With zero-centered weights, `dual_rms_norm` computes the sandwich norm
of Gemma 2 and Gemma 3.

The `_fp8_quant` functions return the E4M3 codes of the BF16 result and need a BF16 `x`. The
argument `fp8_scaling` selects the scales:

- `"tensorwise"` quantizes with the caller's one-element FP32 `fp8_scale` and returns `fp8_scale`
  with the shape `()`. `fused_add_adaptive_rms_norm_fp8_quant` does not take `"tensorwise"`.
- `"rowwise"` returns FP32 scales `[rows, 1]`.
- `"block128"` returns scales `[rows, d / 128]` with the strides `(1, rows)`, FP32 or E8M0 by
  `fp8_scale_dtype`.

The computed scales and the codes follow the rules of [Quantizers](#quantizers). `fp8_gemm` reads
each of these scales as `scale_a`.

The package builds one kernel per combination of the dtypes, the weight, the add, the modulation
and the output. A call that needs another combination raises `ValueError`, which names the built
combinations that differ in one option. For example, `fused_add_rms_norm` with an FP32 `x` takes
only a BF16 weight.

The kernels load each row from global memory into registers. On compute capability 9.0 and up,
`ZIG_CU_KERNELS_RMS_NORM_ENGINE=bulk` selects kernels that first copy the rows into shared memory
with bulk copies (`cp.async.bulk`), and `auto` selects them wherever they run. The default engine
is `registers`. On the H100 and the B300, a call of a bulk kernel took more time than a call of the
`registers` kernel.

## axpby

```python
axpby(alpha, x, beta, y, *, out=None)  # alpha * x + beta * y
axpby_(alpha, x, beta, y)  # writes the result into y and returns y
```

`x` and `y` are BF16 or FP32 tensors of one shape and dtype, without broadcasting or type
promotion. With `a` and `b` the FP32 roundings of `alpha` and `beta`, the result is
`fma(a, x, fp32(b * y))` in FP32, rounded once more for BF16 tensors. When a coefficient rounds
to zero, the call neither reads nor copies the operand of that coefficient and uses +0 in its
place. A NaN in that operand therefore does not reach the result, while `torch.add` would return
NaN.

## Numerics and tolerances

Every bound holds for each element of a result, except the bound for `use_fast_accum=True`, which
compares the largest errors over a result. The 1% counts are statistics over a whole result.

- BF16 GEMM. When every partial sum is exact in FP32, every kernel and split of k gives the bits
  of a reference that applies the epilogue to the exact sum with the PyTorch BF16 CUDA ops. For
  random operands, a BF16 result without activation stays within `δ + 2·ulp(t)` of that reference,
  where `t` is the sum plus the bias, `ulp` is the BF16 spacing at `|t|`, and
  `δ = 2^-22·R·Σ_j |a_j·b_j|`. `R` counts the FP32 roundings that can reach one sum, for example
  `ceil(k / 16) + 1` for a tiled kernel that does not split k. The bounds for activations, gated
  units and FP32 results have the same form. Each epilogue step returns what the PyTorch BF16 CUDA
  op returns, which a test checks for all 65536 BF16 inputs.
- FP8 GEMM. Against the exact product of the dequantized operands, the error stays within `1.01·E`.
  `E` is `γ·Σ_j |A_j·B_j|` plus `2^-8` of each exact value that the epilogue rounds to BF16, and an
  activation multiplies the error before it by its largest slope, at most 1.13. `γ` is `2^-9` for
  codes that the quantizers produce from Gaussian or Student-t data with `k ≤ 65536`, and `2^-6` for
  any finite codes and scales whose products stay in the FP32 range. With `use_fast_accum=True`,
  the largest error over a result stays within twice that of
  `torch._scaled_mm(..., use_fast_accum=True)` plus `2^-8` of the largest exact value.
- The quantizers, the residual streams of the `fused_add_` functions and axpby give the bits of
  their formulas for every input, NaN, infinities and subnormal values included. So do the
  modulation and the FP8 codes and scales of the RMS norms, applied to the kernel's own BF16 norm.
- RMS norm. A BF16 result `y` meets `|y - y_pt| ≤ 3·2^-7·|y_pt| + 2^-126`, where `y_pt` comes from
  the PyTorch sequence of the same roundings, and at most 1% of the results differ from `y_pt` in
  bits. An FP32 result stays within `2^-19` of the exact value, relative. These bounds need finite
  inputs, a row mean square between `2^-60` and `2^60` (or a row of zeros with `eps > 0`), and
  weights, gates, scales and shifts of magnitude at most `2^16`.

NaN and infinities follow IEEE arithmetic, apart from the zero coefficients of axpby. Outside the
activations, which follow the CUDA math library that PyTorch uses, the kernels keep subnormal
values. The tensor cores may treat subnormal GEMM operands as zero.

## Determinism

The same inputs and shapes give the same bits on one device in every call, stream and graph
replay. Batch invariance means that the bits of a row do not change with the row count. The RMS
norms and axpby are batch invariant. The GEMMs are not batch invariant, because the row count
selects the kernel and the split of k, so the last bits of a row can change with `m`. The `tiling`
and `splits` arguments of the BF16 launchers fix the order of the additions.

## Build from source

The kernels compile with a Zig compiler that has the self-hosted PTX backend, and with `ptxas` and
`fatbinary` of a CUDA toolkit that reads PTX ISA 9.0 (CUDA 13.0 or later). The build takes
`fatbinary` from the directory of `ptxas`:

```sh
zig build -Dptxas="$CUDA_HOME/bin/ptxas"
```

The build writes the PTX, one cubin per module and `kernels.manifest` to `zig-out/`, where the
package finds them when Python imports it from `src/`, as the tests do. `ZIG_CU_KERNELS_DIR`
selects another build directory, such as one made with `zig build -p <dir>`. `pip install .` runs
the same build, with the compiler and assembler that `ZIG` and `PTXAS` name, and packages the
cubins. When neither `-Dptxas` nor `PTXAS` names `ptxas`, the build looks for it on the `PATH` and
then in `/usr/local/cuda/bin`.

`-Dsm=sm_80` assembles only the sm_80 machine code, which also runs on 8.6 and 8.9 devices, so a
test run against such a build checks the sm_80 code without an 8.0 GPU. Such a build holds no FP8
machine code. `-Dsource=<name>` builds only the modules compiled from `zig/<name>.zig`, and
`-Dmodule=<name>` builds only the named module. `pip install` passes the comma-separated lists in
`ZIG_CU_KERNELS_SM`, `ZIG_CU_KERNELS_SOURCE` and `ZIG_CU_KERNELS_MODULE` to the build as `-Dsm`,
`-Dsource` and `-Dmodule`, for example `ZIG_CU_KERNELS_SM=sm_80`. `build.zig` records any build
with one of these three options as partial, even when `-Dsm` names every target, and `pip install`
refuses to package a partial build unless `ZIG_CU_KERNELS_ALLOW_PARTIAL=1`. The refusal comes
after the build has written `zig-out/` next to `setup.py`, so a refused build still replaces the
build that the tests find. `-Dkernel-optimize=safe` builds kernels that trap on checked illegal
behavior, such as an integer overflow.

Each build deletes `kernels.manifest` first and writes it after every module, so a failed build
leaves none, and the package then raises `FileNotFoundError`. The deletion runs `rm`, which must be
on the `PATH`. The manifest records a SHA-256 digest of `build.zig` and `zig/*.zig`. When the
parent of the build directory holds `build.zig` and `zig/` with another digest, the first kernel
load or availability query raises `RuntimeError` with the rebuild command.

## Tests

```sh
pytest
```

The tests run from the repository root against the build that the package finds. They check each
function against the bounds of [Numerics and tolerances](#numerics-and-tolerances), and they run the
ops through `torch.library.opcheck`, fake tensors of symbolic sizes, `torch.compile` and CUDA graph
replay. They also check that no call waits for the device. Tests of kernels that the GPU or the
build cannot run skip with the reason. Tests of `benchmarks/timing.py` check the copies of the
operands, the size of the CUDA graphs, the stored rates and the checks that end a benchmark with
status 1.

Every test runs under a watchdog that ends a hung process and prints the stack of every thread.
The time limit of a test follows from the speed-of-light time of its GPU work (see
[Benchmarks](#benchmarks)), and `--time-limit-scale` multiplies every limit, for example under
compute-sanitizer.

## Benchmarks

The scripts in `benchmarks/` print the device time per call as a percentage of up to three limits:

- SoL, the speed-of-light time `max(F / R_peak, D / B_peak)` of `F` floating-point operations and
  `D` bytes of DRAM traffic. `R_peak` is the tensor-core rate of the data sheet at the clock that
  the device reports, and `B_peak` is the DRAM bandwidth.
- The power-capped ceiling, from rates that the GPU sustains at its power limit: a DRAM-bound read,
  a DRAM-bound write and `mma` instructions on register operands.
- The main-loop ceiling applies only to the tiled BF16 GEMM plans. It is a stored rate of the GEMM
  main loop alone, with a text that names where the rate comes from.

`ZIG_CU_KERNELS_CEILINGS` names a JSON file that keeps the measured rates. Each benchmark measures
the DRAM, BF16 or E4M3 rates that it needs and that the file lacks, and adds them to the file.
Without the variable, each process measures the rates that it needs. `python benchmarks/timing.py`
measures all of these rates, and `python benchmarks/benchmark_peak.py` measures the tensor-core
rates. These two scripts keep the highest rate in the file and also store their own result as the
latest rate. A hot GPU sustains lower rates. A result line of a call with tensor-core work therefore
shows two percentages of the power-capped ceiling. The first uses the highest stored rate and stays
comparable between runs. The second uses the latest stored tensor-core rate, which
`benchmark_all.py` measures at the start of its run. No script measures a main-loop rate, and
`python benchmarks/timing.py --main-loop ELEMENT RATE SOURCE` stores one, such as
`--main-loop bf16 1.7e14 "<measurement>"`.

Each call reads and writes its own copy of the operands, and the copies together hold 4 times the
bytes of the L2 cache. The exception is an axpby case whose tensors fit in the L2 cache. That case
reuses one copy, and `benchmark_axpby.py` labels it L2-resident. `benchmark_gemm.py --graph`
replays a CUDA graph of whole cycles through the copies, about 2 ms and at most 1000 calls, which
leaves out the host time of each call. A process exits with status 1 when a result moves its compulsory DRAM bytes faster
than `B_peak`, or more than 5% faster than the measured DRAM rates of the power-capped ceiling.
Such a result shows a measurement error, for example operands that the L2 cache still holds from
an earlier call.

A benchmark process stops after 10 s, so give each process a few cases. `benchmark_all.py` runs
every default case with the PyTorch baselines, a few cases per process, shares one file of rates
among the processes and fails when a process fails. `benchmark_all.py --lock FILE` holds an
exclusive `flock` of `FILE` while each process runs, so other users of the GPU can run between the
processes. A GPU at its power limit runs at a lower clock when it is hot, so compare designs within
one process. Each result line shows the clock, power and temperature of the GPU during its
measurement. With `ZIG_CU_KERNELS_DSE_DIR` set, the scripts also append each result to `dse.csv` in
that directory, a log for design-space exploration (DSE), the comparison of kernel variants by
measurement.

## FP8 operators

These operators multiply FP8 packs. A pack is a pair of tensors, the E4M3 values and the scales
that multiply them. The recipe of a pack says which values share a scale. Each operator name starts
with its recipe, `mxfp8_`, `fp8_b128_` or `fp8_rowwise_`, and the launcher module
`zig_cu_kernels.fp8_mm` calls the recipes `mxfp8`, `fp8_b128_e8m0`, `fp8_b128_fp32` and
`fp8_rowwise`.

Importing `zig_cu_kernels.torch` registers the operators as `torch.ops.zig_cu_kernels.<name>` and
exports each one as `<name>` and `<name>_op`. Each operator has a fake implementation for
`torch.compile` and no backward formula. A call checks the shapes, dtypes and devices of its
arguments, queues its kernels on the current stream and returns without waiting for the device. It
takes its outputs and any temporary buffer from the PyTorch caching allocator.

### Recipes and packs

The formulas below write `A` and `B` for the dequantized packs, the values times their scales. `A`
is `[M, K]` and `B` is `[N, K]`, and both value tensors are row-major and contiguous.

| Recipe | Values that share a scale | Scales of `A` | Scales of `B` | Scale dtype |
|---|---|---|---|---|
| `mxfp8` | 32 consecutive values of a row | `[ceil(M/128), K/128, 32, 16]` | `[N/128, K/128, 32, 16]` | E8M0 byte (`torch.uint8`) |
| `fp8_b128_e8m0`, `fp8_b128_fp32` | 128 consecutive values of a row of `A`, a 128 x 128 tile of `B` | `[ceil(M/128), ceil(K/512), 32, 16]` | `[N/128, 4·ceil(K/512)]` | E8M0 byte or FP32, the same for `A` and `B` |
| `fp8_rowwise` | one row | `[M, 1]` | `[N, 1]` | FP32 |

An E8M0 byte `e` stands for `2^(e - 127)`. A 4-D scale tensor holds atoms of 512 scales, and an atom
covers 128 rows and 4 blocks of each row. The scale of `(row, block)` has the flat index
`atom·512 + (row % 32)·16 + (row % 128 / 32)·4 + block % 4`, with
`atom = (row / 128)·(blocks / 4) + block / 4` and `blocks` the number of blocks of a row rounded up
to a multiple of 4. `tcgen05.cp` copies an atom of this layout to tensor memory unchanged.
`zig_cu_kernels.packs` gives the shapes and the index (`scale_atom_index`). A weight scale tensor of
`fp8_b128` holds `4·ceil(K/512)` entries for each row of tiles, and the quantizer writes 0 into the
entries past `K/128`.

### Products

| Operator | Result | Shapes |
|---|---|---|
| `mxfp8_mm_abt(a, s_a, b, s_b, bias=None, n=None)` | `Y = bf16(A·Bᵀ + bias)`, the first `n` columns | `K`, `N` multiples of 128; `n ≤ N` (default `N`), a multiple of 8; `bias` `[n]` |
| `mxfp8_mm_abt_add(a, s_a, b, s_b, residual)` | `Y = bf16(A·Bᵀ + residual)`, the first `m` rows | `residual` `[m, N]`, `m ≤ M` |
| `mxfp8_mm_abt_low_rank(a, s_a, b, s_b, low, up, bias=None, n=None)` | `Y = bf16(A·Bᵀ + low·upᵀ + bias)` | `low` `[M, r]`, `up` `[n, r]`, `r` 32, 64 or 128 |
| `mxfp8_mm_glu(a, s_a, w1, s_w1, gate_first=False, out_z=False, clamp_limit=None, activation="swiglu", hidden=None)` | `(Z, h, s_h)` | `w1` `[2·Fp, K]`, `Fp` a multiple of 128; `hidden` a multiple of 8, `round128(hidden) = Fp` |
| `mxfp8_mm_qkv_prep(a, s_a, b, s_b, qk_scales, rope, start_elem, eps, n_heads, head_dim, save_qkv)` | `(Q, KV, saved)` | `b` `[3·n_heads·head_dim, K]` |
| `mxfp8_mm_atb(a_cw, a_cw_scales, b_cw, b_cw_scales, n=None)` | `D = bf16(Aᵀ·B)`, the first `n` columns | columnwise packs `[K, M]`, `[K, N]`; `M`, `N` multiples of 128 |
| `fp8_b128_mm_abt(a, s_a, b, s_b)` | `Y = bf16(A·Bᵀ)` | `K`, `N` multiples of 128 |
| `fp8_b128_mm_glu(a, s_a, w1, s_w1, activation, out_z)` | `(Z, h, s_h)` | `w1` `[2·F, K]`, `F` a multiple of 128 |
| `fp8_rowwise_mm_abt(a, s_a, b, s_b, bias)` | `Y = bf16(A·Bᵀ + bias)` | `K` a multiple of 16, `N` a multiple of 8 |
| `fp8_rowwise_mm_glu(a, s_a, w1, s_w1, activation, out_z)` | `(Z, h, s_h)` | `w1` `[2·F, K]`, `F` a multiple of 128 |

`a` needs at least one row. `bias`, `residual`, `low` and `up` are BF16, and every result except the
packs of `h` is BF16. A kernel adds the product, the residual or the low-rank term, and the bias in
FP32, in this order, and rounds the sum to BF16 once.

`mxfp8_mm_atb` computes a weight gradient. `mxfp8_quant_columnwise(x)` returns the columnwise pack
of `x` `[T, C]`, whose values keep the layout of `x` with `C` padded to a multiple of 128 and whose
scales each cover 32 consecutive rows of a column. For `dy` `[T, N]` and `x` `[T, K]`,
`mxfp8_mm_atb(*mxfp8_quant_columnwise(dy), *mxfp8_quant_columnwise(x))` returns `dyᵀ·x` with
`round128(N)` rows and `round128(K)` columns.

### GLU and QKV preparation

The GLU operators compute the BF16 product `Z` of `a` and both halves of `w1`, whose first half `lo`
and second half `hi` have `F` rows each. `gate` is the product with `lo` when `gate_first` is true
and the product with `hi` otherwise, and `up` is the product with the other half.

- SwiGLU (`activation="swiglu"`) computes `h = bf16(up·bf16(gate·sigmoid(gate)))`. A `clamp_limit`
  `L`, rounded to BF16, first clamps `up` to `[-L, L]` and `gate` to at most `L`.
- GEGLU (`"geglu_tanh"`) computes `h = bf16(bf16(gelu_tanh(gate))·up)`. It needs `gate_first=True`
  and no clamp.

The kernels evaluate the sigmoid and the hyperbolic tangent with `ex2.approx` and `rcp.approx`.
`fp8_b128_mm_glu` and `fp8_rowwise_mm_glu` set `gate_first` to `activation == "geglu_tanh"`, so
their `w1` holds `[up | gate]` for SwiGLU and `[gate | up]` for GEGLU, without padding
(`hidden = F`).

`Z` is `[M, 2·hidden]`, the first `hidden` columns of each half, or None when `out_z` is false.
`(h, s_h)` is the pack of `h` in the recipe of the operator, with the rules of
[Pack quantizers](#pack-quantizers): the MXFP8 pack of `h` `[M, Fp]` with the columns from `hidden`
on set to 0, the block128 activation pack of `h` `[M, F]` with the scale dtype of `s_a`, or the
rowwise pack of `h` `[M, F]`.

`mxfp8_mm_qkv_prep` computes the BF16 product `Z` of `a` and `b`, whose rows hold the q, k and v
projections of `n_heads` heads of `head_dim` columns. For each row and each head of q and k, the
kernel sums `bf16(x·x)` over the head in FP32, computes
`c = bf16(qk_scales[head]·rsqrt(sum / head_dim + eps))` and rounds `x·c` to BF16. It then rotates
the interleaved pairs `(2p, 2p + 1)` of the first `w` columns of the head, with
`w = ndims·Hrope·2`, by `cos = rope[row, 2p]` and `sin = rope[row, 2p + 1]` in FP32, and rounds the
pair to BF16. `Q` holds q `[M, n_heads·head_dim]`, `KV` holds k and then v
`[M, 2·n_heads·head_dim]`, and `saved` holds `Z` when `save_qkv` is true and no rows otherwise.
`head_dim` is 64, 128 or 256, `n_heads` is at most 64, and `n_heads·head_dim` is a multiple of 256.
`qk_scales` is FP32 or BF16 `[n_heads]`, `rope` is FP32 `[B, S, ndims, Hrope, 2]` with `B·S = M`,
and `start_elem` is 0.

### GLU backward

`mxfp8_glu_backward(dh, z, gate_first=False, clamp_limit=None, activation="swiglu")` computes the
gradient `dz` `[M, 2F]` of the GLU with respect to its input `z`, the BF16 `Z` `[M, 2F]` of the
forward operator, for the BF16 gradient `dh` `[M, F]` of `h`. `F` is a multiple of 64. The kernel
recomputes `h` from `z` and returns `(dz, s_dz, dz_cw, s_dz_cw, h_cw, s_h_cw)`, the packs
`mxfp8_quant(dz)`, `mxfp8_quant_columnwise(dz)` and `mxfp8_quant_columnwise(h)` of the BF16 `dz` and
`h`. The columnwise pack of `h` has `round128(F)` columns, and its pad columns hold 0. The
columnwise packs are the operands of `mxfp8_mm_atb`.

`mxfp8_glu_backward_bf16(dh, z, gate_first=False, out_h=False, out_q=False, clamp_limit=None,
activation="swiglu")` returns `(dz, dz_q, s_dz, h)`. `dz` is the BF16 gradient, `(dz_q, s_dz)` is
`mxfp8_quant(dz)` when `out_q` is true, `h` is the BF16 `h` `[M, F]` when `out_h` is true, and an
output that is not asked for is None.

`gate`, `up`, `gate_first`, `clamp_limit` and `activation` follow the forward operator.

- SwiGLU computes `s = sigmoid(gate)` and `silu = gate·s` in FP32, then
  `dz_up = bf16(dh·silu)`, `dz_gate = bf16(dh·up·(silu·(1 - s) + s))` and `h = bf16(up·silu)`.
  The gradient of a value that the clamp changed is 0. This `h` keeps `silu` in FP32, so it can
  differ from the `h` of the forward operator, which rounds `silu` to BF16.
- GEGLU computes `g = bf16(gelu_tanh(gate))`, then `dz_up = bf16(dh·g)`,
  `dz_gate = bf16(gelu_tanh'(gate)·bf16(dh·up))` and `h = bf16(up·g)`.

### Pack quantizers

| Operator | Input | Result |
|---|---|---|
| `mxfp8_quant(x)` | `[rows, K]` or `[L, rows, K]` | MXFP8 pack with values `[rows, round128(K)]`; a 3-D input pads the rows of each matrix to a multiple of 128 |
| `mxfp8_quant_(x, values, scales)` | as `mxfp8_quant` | writes the MXFP8 pack into `values` and `scales` |
| `mxfp8_quant_columnwise(x)` | `[rows, K]` | columnwise MXFP8 pack, the operand of `mxfp8_mm_atb` |
| `mxfp8_quant_dual(x)` | `[rows, K]` | the MXFP8 pack and the columnwise pack from one read of `x` |
| `fp8_b128_quant(x, e8m0)` | `[rows, K]`, `K` a multiple of 128 | block128 activation pack |
| `fp8_b128_quant_weight(w, e8m0)` | `[N, K]`, both multiples of 128 | block128 weight pack |
| `fp8_rowwise_quant(x)` | `[..., K]` | rowwise pack with scales `[..., 1]` |

The inputs are contiguous BF16 or FP32 CUDA tensors that start at a multiple of 16 bytes, with rows
of a multiple of 16 values. Here amax is the largest magnitude among the values that share a scale,
and `1/448` is that quotient rounded to FP32.

- An `mxfp8` or `fp8_b128_e8m0` scale is the least power of two at or above `amax·(1/448)`, or 1
  when amax is 0, and each value becomes `x` divided by its scale.
- An `fp8_b128_fp32` scale is `max(amax·(1/448), 1e-30)`, and each value becomes `x` times the
  correctly rounded FP32 reciprocal of its scale.
- An `fp8_rowwise` scale is `max(amax, 1e-12)·(1/448)`, with `1e-12` rounded to the dtype of the
  input and the scale rounded to BF16 for a BF16 input. Each value becomes the correctly rounded
  FP32 quotient of `x` and its scale. These are the packs of torchao `PerRow` on a GPU.

Each E4M3 value is the FP32 result rounded to the nearest E4M3 value with ties to even, and a result
of magnitude above 448 becomes 448 with its sign.

### GPUs

`zig_cu_kernels.fp8_mm` keeps a table of the kernels for each GPU architecture, recipe and
operation. The table below lists the paths of each operator by compute capability.

- A tiled kernel computes the output in tiles.
- A decode kernel takes a product of 1 to 128 rows of `a` with `K` and `N` multiples of 128. The
  kernel splits the weight into units of 128 rows by 128 values of `K` and gives each work group, at
  most one per SM, an equal share of the units (stream-K). A second kernel adds the partial sums of
  a shared tile in a fixed order. When no tile is shared, the second kernel does not run.
- A fused GLU or QKV kernel computes `h` or `Q` and `KV` from the FP32 sums of its tile.
- A composed chain writes `Z` with an `mm_abt` call and computes `h` or `Q` and `KV` from `Z` with a
  second kernel. The `mm_abt` call takes the decode kernel when the product qualifies.
- "no" means that the operator raises `NotImplementedError`, which names the missing kernel.

| Operator | 8.9 | 9.0 | 10.x | 12.x |
|---|---|---|---|---|
| `mxfp8_mm_abt`, `mxfp8_mm_abt_add` | tiled | no | tiled, decode | tiled, decode |
| `mxfp8_mm_abt_low_rank` | tiled | no | tiled | tiled |
| `mxfp8_mm_glu` | fused without `Z`, composed with `Z` | no | fused | fused |
| `mxfp8_mm_qkv_prep` | composed | no | composed | fused for `head_dim` 128, composed otherwise |
| `mxfp8_mm_atb` | no | no | tiled | no |
| `fp8_b128_mm_abt` with E8M0 scales | tiled | tiled, decode | tiled, decode | tiled, decode |
| `fp8_b128_mm_abt` with FP32 scales | tiled | tiled, decode | no | tiled, decode |
| `fp8_b128_mm_glu` with E8M0 scales | fused without `Z`, composed with `Z` | fused | composed | fused |
| `fp8_b128_mm_glu` with FP32 scales | fused without `Z`, composed with `Z` | fused | no | fused |
| `fp8_rowwise_mm_abt` | tiled | tiled, decode | tiled, decode | tiled, decode |
| `fp8_rowwise_mm_glu` | fused without `Z`, composed with `Z` | composed | composed | fused without `Z`, composed with `Z` |
| `mxfp8_glu_backward`, `mxfp8_glu_backward_bf16` | yes | yes | yes | yes |
| pack quantizers | yes | yes | yes | yes |

- On 8.9, a fused GLU kernel runs only when the product selects the 128 x 256 tile (128 x 128 for
  `mxfp8`), and other products take the composed chain.
- The fused rowwise GLU kernels write BF16 `h`, and a second kernel quantizes its rows, because a
  row scale needs the whole row.
- On 10.x, a product of 65 to 128 rows takes the tiled kernel when the weight has at least
  `0.85·SMs` tiles of 128 rows.
- The decode launcher of 12.x starts one work group per SM. On 10.x it starts the largest multiple
  of the weight tile count that fits in the SMs, so that each work group stays inside one tile, and
  one work group per SM when that multiple would leave more than 15% of the SMs idle. On 9.0 it
  picks a work-group count from 90% of the SMs to all of them. It takes the count with the least
  modeled time. The model takes the longer of the time to stream the weight and the time of the
  busiest work group's units, and adds the time to write and read the partial sums.
- On 9.0, `fp8_rowwise_mm_abt` adds the product of each 128 values of `K` to FP32 sums, as the
  block128 kernels do. The launcher `mm_sm90.fp8_rowwise_mm_abt_fast` accumulates over all of `K`
  in the tensor cores, as `torch._scaled_mm(..., use_fast_accum=True)` does.
- On 9.0, the package builds a kernel of `fp8_rowwise_mm_abt` without a bias with 192 x 128 tiles
  and three consumer warpgroups. The launcher selects it only when `mm_sm90.THREE_CONSUMERS_MIN_K`
  is an integer, `K` is at least that integer, and each cluster then walks no more tiles than with
  the default 128 x 192 tiles. The default of `THREE_CONSUMERS_MIN_K` is `None`, so the launcher
  selects a kernel with two consumer warpgroups.

The 9.0 kernels need compute capability 9.0 exactly (`sm_90a`), and the 10.x and 12.x kernels run
on every device of their family (`sm_100f`, `sm_120f`). A build without `-Dsm` holds the machine
code of every path in the table for each of these GPUs (see [Requirements](#requirements)).

### FP8 numerics

- Each element of `Y` or `D` stays within `2^-9·Σ_k |A_ik·B_jk| + 2^-8·|exact|` of the exact
  value, which is the FP64 product of the dequantized packs plus the BF16 terms of the epilogue. On
  9.0, the test of `fp8_rowwise_mm_abt` checks the relative Frobenius error
  `‖Y - exact‖ ≤ 2^-8·‖exact‖` instead.
- When every partial sum is exact in FP32, for example for small integer values with power-of-two
  scales, every kernel, tile and split gives the bits of the exact sum rounded once to BF16 after
  the epilogue terms. The decode kernels give the bits of the tiled kernels, and two runs give the
  same bits.
- The fused GLU and QKV kernels give the bits of the composed chain, `Z` included.
- The GLU backward operators give the bits of a PyTorch reference wherever the reference gives the
  same value at both ends of the error bounds of `ex2.approx` and `rcp.approx`. The SwiGLU backward
  flushes subnormal operands and results of its FP32 steps to zero, and the GEGLU backward does not
  flush its products.
- The GLU pack `h` matches a PyTorch reference bit for bit in every quantization block where the
  reference gives the same block at both ends of the error bounds of `ex2.approx` and `rcp.approx`.
  `Q` and the k half of `KV` stay within `2^-6` of the largest magnitude of their row and head, and
  at most 0.1% of their elements differ from all three references that evaluate `rsqrt` at the ends
  and the middle of the error bound of `rsqrt.approx`. `saved` and the v half of `KV` equal `Z`.
- The pack quantizers give the bits of the PyTorch references in `tests/pack_references.py` for
  every tested input, which includes every finite BF16 value, values at the saturation limit, E4M3
  rounding ties and magnitudes from 2^-148 to 2^124. No test covers NaN or infinite inputs.

### FP8 tests and benchmark

`pytest` runs the FP8 tests with the rest of the suite, and tests of kernels that the GPU or the
build lacks skip with the reason. `tests/test_fp8_mm.py` checks the operators: contracts, fake
results, `opcheck`, the error bound, the bitwise epilogues and the GLU and QKV chains.
`tests/test_fp8_pack.py` and `tests/test_packs.py` check the quantizers and the layouts, and
`tests/test_glu_bwd.py` checks the GLU backward. The files
`tests/test_mm_sm90.py`, `test_mm_sm120.py`, `test_mm_sm100.py`, `test_fp8_mm_glu_sm89.py` and
`test_fp8_b128_mm_glu.py` hold the bitwise tests of the kernels of one GPU generation, and
`tests/test_decode*.py` test the decode schedule and kernels.

`benchmarks/benchmark_fp8_mm.py` times `mm_abt` and `mm_glu` of each recipe and the
`torch._scaled_mm` references on one GPU:

```sh
python benchmarks/benchmark_fp8_mm.py --ops mm_abt mm_glu --recipes mxfp8 fp8_rowwise \
    --shapes gemma_down:2048 --json results.json
```

A shape is `MxNxK`, the name of a production shape such as `gemma_qkv`, which stands for each of
its row counts, or a name with one row count such as `gemma_qkv:2048`. `--help` lists the names.
The script quantizes random normal inputs with the pack quantizers before it times anything. Each
result line shows the time, TFLOP/s, percent of SoL, percent of each matching reference, the
relative error against the FP32 product of the dequantized packs, and the SM clock and board power.

The SoL of this script is `max(F / P, D / W)`, where `F = 2·M·N·K` and `D` counts the packs and
outputs once. `P` is the FP8 tensor-core rate that peak-rate kernels sustained and `W` the copy
rate, both measured on each GPU model (the table `DEVICES` of the script). `mxfp8` is compared with
torch MXFP8, the `fp8_b128` recipes with torch blockwise scaling, and either falls back to torch
rowwise with fast accumulation where the GPU lacks that reference. `fp8_rowwise` is compared with
torch rowwise with and without fast accumulation. A reference whose relative error exceeds 2^-8 is
marked as disagreeing. An `mm_glu` result is compared with the reference GEMM of the same shape,
which leaves out the GLU and the quantization of its output.

Each measurement lasts about 300 ms, and the script times products of at most 512 rows as CUDA
graph replays. The script ends its measurements `--budget` seconds after the process started (9 by
default) and prints each case that it skipped, so a process has time for one shape with every
recipe.
