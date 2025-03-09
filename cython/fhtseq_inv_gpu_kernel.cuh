// fhtseq_inv_gpu_kernel.cuh
#ifndef FHTSEQ_INV_GPU_KERNEL_CUH
#define FHTSEQ_INV_GPU_KERNEL_CUH

#include <cuda_runtime.h>

extern "C" void fhtseq_inv_gpu_kernel(float *x, int N, int L);

#endif // FHTSEQ_INV_GPU_KERNEL_CUH
