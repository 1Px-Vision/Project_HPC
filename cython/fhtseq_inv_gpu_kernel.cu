// fhtseq_inv_gpu_kernel.cu
#include "fhtseq_inv_gpu_kernel.cuh"

__global__ void fhtseq_inv_gpu(float *x, int N, int L) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = blockDim.x * gridDim.x;

    __shared__ float sdata[32];

    for (int pos = tid; pos < N; pos += threadNum) {
        sdata[threadIdx.x] = x[pos];
        __syncthreads();

        int k1 = N;
        int k2 = 1;
        int k3 = N >> 1;

        for (int i1 = 0; i1 < L; ++i1) {
            int L1 = 0;
            for (int i2 = 0; i2 < k2; ++i2) {
                bool isOdd = i2 & 1;
                for (int i3 = 0; i3 < k3; ++i3) {
                    int ii = i3 + L1;
                    int jj = ii + k3;

                    float temp1 = sdata[ii];
                    float temp2 = sdata[jj];

                    float add = temp1 + temp2;
                    float sub = temp1 - temp2;

                    sdata[ii] = isOdd ? sub : add;
                    sdata[jj] = isOdd ? add : sub;
                }
                L1 += k1;
            }
            k1 >>= 1;
            k2 <<= 1;
            k3 >>= 1;
        }
        x[pos] = sdata[threadIdx.x] / N;
        __syncthreads();
    }
}

extern "C" void fhtseq_inv_gpu_kernel(float *x, int N, int L) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    fhtseq_inv_gpu<<<grid_size, block_size>>>(x, N, L);
}
