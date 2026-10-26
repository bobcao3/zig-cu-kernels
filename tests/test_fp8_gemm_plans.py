import pytest

from zig_cu_kernels import fp8_gemm, runtime


@pytest.mark.parametrize("family", ("rowwise", "block128"))
@pytest.mark.parametrize("sms", (64, 128, 160))
def test_store_keeps_the_faster_tiling_for_a_partial_row_tile(family, sms):
    _, entries = runtime.manifest()
    if any(fp8_gemm.gemm_name(family, "store", tiling) not in entries for tiling in fp8_gemm.TILINGS):
        pytest.skip("the FP8 GEMM kernels are not built")
    choice = fp8_gemm.choose(family, "store", 1000, 1024, 2048, sms, 960e9, False)
    assert choice == fp8_gemm.Choice("128x128", 64, 64)
