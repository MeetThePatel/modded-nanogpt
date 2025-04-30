#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <c10/util/Exception.h>
#include <torch/extension.h>

torch::Tensor newton_schulz_cuda(const torch::Tensor &G, const int ns_steps);

namespace {

// Definitions for `args` and `r_args`.
constexpr int DEPTH = 3;
constexpr int PARAM_IDX = 0;
constexpr int GRAD_IDX = 1;
constexpr int MOMENTUM_BUFFER_IDX = 2;

template <typename scalar_t>
C10_DEVICE __forceinline__ void muon_math_prologue(scalar_t r_args[DEPTH][at::native::kILP], const double momentum, const bool is_first_step) {
    using opmath_t = at::opmath_type<scalar_t>;

#pragma unroll
    for (int ii = 0; ii < at::native::kILP; ++ii) {
        opmath_t p = static_cast<opmath_t>(r_args[PARAM_IDX][ii]);
        opmath_t g = static_cast<opmath_t>(r_args[GRAD_IDX][ii]);

        const opmath_t momentum_buffer = is_first_step ? g : (momentum * static_cast<opmath_t>(r_args[MOMENTUM_BUFFER_IDX][ii]) + g);
        r_args[MOMENTUM_BUFFER_IDX][ii] = momentum_buffer;

        g = g + momentum * momentum_buffer; // Nesterov
        r_args[GRAD_IDX][ii] = g;
    }
}

template <typename scalar_t>
C10_DEVICE __forceinline__ void muon_math_epilogue(scalar_t r_args[DEPTH][at::native::kILP], const double lr) {
    using opmath_t = at::opmath_type<scalar_t>;

#pragma unroll
    for (int ii = 0; ii < at::native::kILP; ++ii) {
        opmath_t p = static_cast<opmath_t>(r_args[PARAM_IDX][ii]);
        opmath_t g = static_cast<opmath_t>(r_args[GRAD_IDX][ii]);

        p -= lr * g;
        r_args[PARAM_IDX][ii] = p;
    }
}

template <typename scalar_t>
struct FusedMuonMathPrologueFunctor {

    C10_DEVICE __forceinline__ void operator()(const int chunk_size, at::native::TensorListMetadata<DEPTH> &tl, const double momentum,
                                               const bool is_first_step) {
        const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
        const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

        scalar_t *args[DEPTH];
        scalar_t r_args[DEPTH][at::native::kILP];
        const auto all_aligned{at::native::init_args<DEPTH>(args, tl, chunk_idx, chunk_size, tensor_loc)};
        const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

        const bool use_faster_load_store = (n % at::native::kILP == 0) && (chunk_size & at::native::kILP == 0) && all_aligned;
        if (use_faster_load_store) {
            for (auto i_start = threadIdx.x; i_start * at::native::kILP < n && i_start * at::native::kILP < chunk_size; i_start += blockDim.x) {
#pragma unroll
                for (auto i = 0; i < DEPTH; ++i) {
                    at::native::load_store(r_args[i], args[i], 0, i_start);
                }

                muon_math_prologue<scalar_t>(r_args, momentum, is_first_step);

                at::native::load_store(args[GRAD_IDX], r_args[GRAD_IDX], i_start, 0);
                at::native::load_store(args[MOMENTUM_BUFFER_IDX], r_args[MOMENTUM_BUFFER_IDX], i_start, 0);
            }
        } else {
            for (auto i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * at::native::kILP) {
                at::native::load_args<DEPTH>(r_args, args, i_start, chunk_size, n);

                muon_math_prologue<scalar_t>(r_args, momentum, is_first_step);

                at::native::store_args(args[GRAD_IDX], r_args[GRAD_IDX], i_start, chunk_size, n);
                at::native::store_args(args[MOMENTUM_BUFFER_IDX], r_args[MOMENTUM_BUFFER_IDX], i_start, chunk_size, n);
            }
        }
    }
};

template <typename scalar_t>
struct FusedMuonMathEpilogueFunctor {

    C10_DEVICE __forceinline__ void operator()(const int chunk_size, at::native::TensorListMetadata<DEPTH> &tl, const double lr) {
        const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
        const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

        scalar_t *args[DEPTH];
        scalar_t r_args[DEPTH][at::native::kILP];
        const auto all_aligned{at::native::init_args<DEPTH>(args, tl, chunk_idx, chunk_size, tensor_loc)};
        const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

        const bool use_faster_load_store = (n % at::native::kILP == 0) && (chunk_size & at::native::kILP == 0) && all_aligned;
        if (use_faster_load_store) {

            for (auto i_start = threadIdx.x; i_start * at::native::kILP < n && i_start * at::native::kILP < chunk_size; i_start += blockDim.x) {
                at::native::load_store(r_args[PARAM_IDX], args[PARAM_IDX], 0, i_start);
                at::native::load_store(r_args[GRAD_IDX], args[GRAD_IDX], 0, i_start);

                muon_math_epilogue<scalar_t>(r_args, lr);

                at::native::load_store(args[PARAM_IDX], r_args[PARAM_IDX], i_start, 0);
                at::native::load_store(args[GRAD_IDX], r_args[GRAD_IDX], i_start, 0);
            }
        } else {
            for (auto i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * at::native::kILP) {
                at::native::load_args<DEPTH>(r_args, args, i_start, chunk_size, n);

                muon_math_epilogue<scalar_t>(r_args, lr);

                at::native::store_args(args[PARAM_IDX], r_args[PARAM_IDX], i_start, chunk_size, n);
                at::native::store_args(args[GRAD_IDX], r_args[GRAD_IDX], i_start, chunk_size, n);
            }
        }
    }
};

} // namespace

void _fused_muon_kernel_cuda_(at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list, const double momentum,
                              const double lr, const int ns_steps, const bool is_first_step) {
    TORCH_CHECK_GT(momentum, 0.0);
    TORCH_CHECK_LE(momentum, 1.0);
    TORCH_CHECK(!momentum_buffer_list.empty(), "momentum buffer cannot be empty.");
    TORCH_CHECK_GE(lr, 0.0);
    TORCH_CHECK_GT(ns_steps, 0);

    TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads, momentum_buffer_list}));

    std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec(), momentum_buffer_list.vec()};

    AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, params[0].scalar_type(), "fused_muon_prologue_kernel_cuda", [&]() {
        at::native::multi_tensor_apply<DEPTH>(tensor_lists, FusedMuonMathPrologueFunctor<scalar_t>(), momentum, is_first_step);
    });

    for (auto grad : grads) {
        grad = newton_schulz_cuda(grad, ns_steps);
    }

    AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, params[0].scalar_type(), "fused_muon_epilogue_kernel_cuda",
                                   [&]() { at::native::multi_tensor_apply<DEPTH>(tensor_lists, FusedMuonMathEpilogueFunctor<scalar_t>(), lr); });
}

// This is just for consistency with Torch's API that accepts lr as a tensor.
void _fused_muon_kernel_cuda_(at::Tensor params, at::TensorList grads, at::TensorList momentum_buffer_list, const double momentum,
                              const at::Tensor &lr, const int ns_steps, const bool is_first_step) {
    _fused_muon_kernel_cuda_(params, grads, momentum_buffer_list, momentum, lr.item<double>(), ns_steps, is_first_step);
}
