#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/layout.h>

using ElementInput = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using Arch = cutlass::arch::Sm89;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using Gemm_XXt = cutlass::gemm::device::GemmUniversalAdapter<cutlass::gemm::kernel::DefaultGemmUniversal<
    ElementInput, LayoutInputA, cutlass::ComplexTransform::kNone, ElementInput, LayoutInputB, cutlass::ComplexTransform::kConjugate, ElementOutput,
    LayoutOutput, ElementAccumulator, OperatorClass, Arch, cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementOutput>,
    Swizzle, 2>>;
