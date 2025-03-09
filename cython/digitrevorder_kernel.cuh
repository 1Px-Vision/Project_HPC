// digitrevorder_kernel.cuh
#ifndef DIGITREVORDER_KERNEL_CUH
#define DIGITREVORDER_KERNEL_CUH

#include <cuda_runtime.h>

extern "C" void digitrevorder_kernel(int *x, int *vec, int *result, int N, int L, int base);

#endif // DIGITREVORDER_KERNEL_CUH
