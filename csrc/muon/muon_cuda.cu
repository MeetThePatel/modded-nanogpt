#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <c10/util/Exception.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t, int depth>
C10_DEVICE __forceinline__ void muon_math(scalar_t r_args[depth][at::native::kILP], const double momentum, const float *lr_ptr, const double lr,
                                          const bool nesterov, const bool is_first_step) {
    using opmath_t = at::opmath_type<scalar_t>;
    const double double_lr = lr_ptr != nullptr ? *lr_ptr : lr;

#pragma unroll
    for (int ii = 0; ii < at::native::kILP, ++ii) {
        auto p = static_cast<opmath_t>(r_args[0][ii]);
        auto g = static_cast<opmath_t>(r_args[1][ii]);

        if (depth > 2) {
            const auto momentum_buffer = is_first_step ? g : (momentum * static_cast<opmath_t>(r_args[2][ii]) + g);
            r_args[2][ii] = momentum_buffer;

            if (nesterov) {
                g = g + momentum * momentum_buffer;
            } else {
                g = momentum_buffer;
            }
        }
        p -= double_lr * g;
        r_args[0][ii] = p;
    }
}

template <typename scalar_t, int depth>
struct FusedMuonMathFunctor {

    C10_DEVICE __forceinline__ void operator()(const int chunk_size, at::native::TensorListMetadata<depth> &tl, const double momentum,
                                               const float *lr_ptr, const double lr, const bool nesterov, const bool is_first_step) {
        const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
        const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

        scalar_t *args[depth];
        scalar_t *r_args[depth][at::native::kILP];
        const auto all_aligned{at::native::init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
        const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

        const bool use_faster_load_store = (n % at::native::kILP == 0) && (chunk_size & at::native::kILP == 0) && all_aligned;
        if (use_faster_load_store) {
            for (auto i_start = threadIdx.x; i_start * at::native::kILP < n && i_start * at::native::kILP < chunk_size; i_start += blockDim.x) {
#pragma unroll
                for (auto i = 0; i < depth; ++i) {
                    at::native::load_store(r_args[i], args[i], 0, i_start);
                }
                muon_math<scalar_t, depth>(r_args, momentum, lr_ptr, lr, nesterov, is_first_step);
                at::native::load_store(args[0], r_args[0], i_start, 0);
                if (depth > 2) {
                    at::native::load_store(args[2], r_args[2], i_start, 0);
                }
            }
        } else {
            for (auto i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * at::native::kILP) {
                at::native::load_args<depth>(r_args, args, i_start, chunk_size, n);
                muon_math<scalar_t, depth>(r_args, momentum, lr_ptr, lr, nesterov, is_first_step);
                at::native::store_args(args[0], r_args[0], i_start, chunk_size, n);
                if (depth > 2) {
                    at::native::store_args(args[2], r_args[2], i_start, chunk_size, n);
                }
            }
        }
    }
};

void _fused_muon_with_momentum_kernel_cuda_(at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list, const double momentum,
                                            const double lr, const bool nesterov, const bool is_first_step) {
    TORCH_CHECK_GT(momentum, 0);
    TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads, momentum_buffer_list}));

    float *lr_ptr = nullptr;

    std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec(), momentum_buffer_list.vec()};
    AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, params[0].scalar_type(), "fused_muon_with_momentum_kernel_cuda", [&]() {
        at::native::multi_tensor_apply<3>(tensor_lists, FusedMuonMathFunctor<scalar_t, 3>(), momentum, lr_ptr, lr, nesterov, is_first_step);
    });
}

void _fused_muon_with_momentum_kernel_cuda_(at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list, const double momentum,
                                            const at::Tensor &lr, const bool nesterov, const bool is_first_step) {

    if (lr.is_cpu()) {
        _fused_muon_with_momentum_kernel_cuda_(params, grads, momentum_buffer_list, momentum, lr.item<double>(), nesterov, is_first_step);
        return;
    }

    TORCH_CHECK_GT(momentum, 0);
    TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads, momentum_buffer_list}));

    TORCH_CHECK(lr.device() == params[0].device(), "lr must be on the same GPU device as the params.");

    std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec(), momentum_buffer_list.vec()};

    AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, params[0].scalar_type(), "fused_muon_with_momentum_kernel_cuda", [&]() {
        at::native::multi_tensor_apply<3>(tensor_lists, FusedMuonMathFunctor<scalar_t, 3>(), momentum, lr.data_ptr<double>(), 1.0, nesterov,
                                          is_first_step);
    });
}

} // namespace

void _fused_muon_kernel_cuda_(at::TensorList params, at::TensorList grads, at::TensorList momentum_buffer_list, const double momentum,
                              const double lr, const bool nesterov, const bool is_first_step) {
    TORCH_CHECK(!momentum_buffer_list.empty(), "momentum buffer cannot be empty.");
    TORCH_CHECK_GT(momentum, 0);
    TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads, momentum_buffer_list}));

    float *lr_ptr = nullptr;

    std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec(), momentum_buffer_list.vec()};
    AT_DISPATCH_FLOATING_TYPES_AND(at::kBFloat16, params[0].scalar_type(), "fused_muon_with_momentum_kernel_cuda", [&]() {
        at::native::multi_tensor_apply<3>(tensor_lists, FusedMuonMathFunctor<scalar_t, 3>(), momentum, lr_ptr, lr, nesterov, is_first_step);
    });
    // _fused_muon_with_momentum_kernel_cuda_(params, grads, momentum_buffer_list, momentum, lr, nesterov, is_first_step);
}
