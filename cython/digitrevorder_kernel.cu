// digitrevorder_kernel.cu
#include "digitrevorder_kernel.cuh"

#define warpSize 32

__global__ void digitrevorder_gpu(int *x, int *vec, int *result, int N, int L, int base) {
    __shared__ int s_vec[32];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        s_vec[i] = vec[i];
    }
    __syncthreads();

    if (idx < ((N + warpSize - 1) / warpSize) * warpSize) {
        int res = 0;
        if (idx < N) {
            int temp = x[idx];
            #pragma unroll
            for (int k = L - 1; k >= 0; --k) {
                int div = temp / s_vec[k];
                res += div * s_vec[L - 1 - k];
                temp -= div * s_vec[k];
            }
            result[idx] = res;
        } else {
            result[idx] = 0;
        }
    }
}

extern "C" void digitrevorder_kernel(int *x, int *vec, int *result, int N, int L, int base) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    digitrevorder_gpu<<<grid_size, block_size>>>(x, vec, result, N, L, base);
}
