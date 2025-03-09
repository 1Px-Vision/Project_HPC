# cython_CUDA_FHSPI.pyx

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cdef extern from "cuda_runtime.h":
    void cudaMalloc(void **devPtr, size_t size)
    void cudaFree(void *devPtr)
    void cudaMemcpy(void *dst, void *src, size_t count, int kind)

cdef extern from "digitrevorder_kernel.cuh":
    void digitrevorder_kernel_launcher(int *x, int *vec, int *result, int N, int L, int base)

cdef extern from "fhtseq_inv_gpu_kernel.cuh":
    void fhtseq_inv_gpu_launcher(float *x, int N, int L)

# Constants for cudaMemcpy
cdef int cudaMemcpyHostToDevice = 1
cdef int cudaMemcpyDeviceToHost = 2

def digitrevorder(np.ndarray[np.int32_t, ndim=1] x, int base):
    cdef int N = x.shape[0]
    cdef int rem = N, L = 0
    while rem >= base:
        rem //= base
        L += 1
    if rem != 1:
        raise ValueError("Length must be a power of 'base'.")

    cdef np.ndarray[np.int32_t, ndim=1] vec = np.array([base ** n for n in range(L)], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] result = np.zeros(N, dtype=np.int32)

    cdef int *d_x, *d_vec, *d_result
    cudaMalloc(&d_x, N * sizeof(int))
    cudaMalloc(&d_vec, L * sizeof(int))
    cudaMalloc(&d_result, N * sizeof(int))

    cudaMemcpy(d_x, &x[0], N * sizeof(int), cudaMemcpyHostToDevice)
    cudaMemcpy(d_vec, &vec[0], L * sizeof(int), cudaMemcpyHostToDevice)

    digitrevorder_kernel_launcher(d_x, d_vec, d_result, N, L, base)

    cudaMemcpy(&result[0], d_result, N * sizeof(int), cudaMemcpyDeviceToHost)

    cudaFree(d_x)
    cudaFree(d_vec)
    cudaFree(d_result)

    return result

def fhtseq_inv(np.ndarray[np.float32_t, ndim=1] data):
    cdef int N = data.shape[0]
    cdef int L = int(np.log2(N))
    if 2 ** L != N:
        raise ValueError("Length must be a power of 2.")

    data_bitrev = digitrevorder(np.arange(N, dtype=np.int32), 2)
    cdef np.ndarray[np.float32_t, ndim=1] reordered_data = data[data_bitrev]

    cdef float *d_x
    cudaMalloc(&d_x, N * sizeof(float))
    cudaMemcpy(d_x, &reordered_data[0], N * sizeof(float), cudaMemcpyHostToDevice)

    fhtseq_inv_gpu_launcher(d_x, N, L)

    cdef np.ndarray[np.float32_t, ndim=1] result = np.zeros(N, dtype=np.float32)
    cudaMemcpy(&result[0], d_x, N * sizeof(float), cudaMemcpyDeviceToHost)

    cudaFree(d_x)

    return result

def fwht2d(np.ndarray[np.float32_t, ndim=2] xx):
    cdef int N = xx.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] xx1 = np.zeros((N, N), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] xx_out = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        xx1[i, :] = fhtseq_inv(xx[i, :])

    for j in range(N):
        xx_out[:, j] = fhtseq_inv(xx1[:, j])

    return xx_out

def getHSPIReconstruction_GPU(np.ndarray[np.float32_t, ndim=3] dataMat, int nStep):
    if nStep != 2:
        raise ValueError("Only nStep=2 is implemented.")

    cdef np.ndarray[np.float32_t, ndim=2] spec = dataMat[:, :, 0] - dataMat[:, :, 1]
    cdef np.ndarray[np.float32_t, ndim=2] img = fwht2d(spec)

    return img, spec
