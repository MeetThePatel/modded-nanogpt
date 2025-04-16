#include <torch/extension.h>

template<typename scalar_t>
__global__ void
compute_B(const scalar_t* A_d,
          const scalar_t* A2_d,
          float* B_d,
          int dim,
          float b,
          float c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dim * dim;
  if (idx < total) {
    B_d[idx] = NS_b * A_d[idx] + NS_c * A2_d[idx];
  }
}

template<typename scalar_t>
__global__ void
newton_schulz_fused_kernel(const scalar_t* G_d,
                           scalar_t* X_d,
                           int rows,
                           int cols)
{
  extern __shared__ float shared_mem[];
  float* A = shared_mem;
  float* A2 = A + (rows * rows);
  float* B = A2 + (rows * rows);
  float* temp = B + (rows * rows);

  int threadid = threadIdx.x;
  int num_threads = blockDim.x;

  for (int index = threadid; index < rows * cols; index += num_threads) {
    X_d[index] = G_d[index];
  }
  __syncthreads();
}

template<typename scalar_t, int depth>
struct MuonFunctor
{}