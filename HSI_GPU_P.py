import numpy as np
import math
import cv2
from numpy import r_
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import time
from time import time
import scipy.io
from PIL import Image
import torch
import threading
from multiprocessing import Process, shared_memory
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
from pycuda.elementwise import ElementwiseKernel

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context

#define warpSize 32
mod = SourceModule("""
__global__ void digitrevorder_kernel(int *x, int *vec, int *result, int N, int L, int base) {
 __shared__ int s_vec[32]; // Use dynamic shared memory based on L

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Ensure L is a multiple of warpSize for coalesced memory accesses
    // Assuming L is a multiple of blockDim.x
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        s_vec[i] = vec[i];
    }
    __syncthreads(); // Ensure all the required vec values are loaded into shared memory

    // Padding N to avoid warp divergence
    if (idx < ((N + warpSize - 1) / warpSize) * warpSize) {
        int res = 0;
        if (idx < N) {
            int temp = x[idx];
            // Manually unroll the loop if L is a known compile-time constant
            #pragma unroll
            for (int k = L - 1; k >= 0; --k) {
                // Compute once and store to minimize recomputation
                int div = temp / s_vec[k];
                res += div * s_vec[L - 1 - k];
                temp -= div * s_vec[k];
            }
            result[idx] = res;
        }
        else {
            // Handle boundary conditions for warps that deal with array end
            result[idx] = 0; // Assuming 0 is a neutral value for the result
        }
    }
}
""")

digitrevorder_kernel_2 = mod.get_function("digitrevorder_kernel")

def bitrevorder_cuda(x):
    temp_x=np.arange(0,len(x))
    temp_y=digitrevorder_cuda(temp_x,2)
    return x[temp_y]

def digitrevorder_cuda(x, base):
    x = np.asarray(x, dtype=np.int32)
    rem = N = len(x)
    L = 0
    while rem >= base:
        rem //= base
        L += 1
    if rem != 1:
        raise ValueError("Length of data must be power of base.")

    vec = np.array([base ** n for n in range(L)], dtype=np.int32)
    result = np.zeros_like(x)

    #digitrevorder_kernel_2 = mod.get_function("digitrevorder_kernel")

    block_size = 256
    grid_size = int(np.ceil(N / block_size))

    digitrevorder_kernel_2(
        cuda.In(x), cuda.In(vec), cuda.Out(result),
        np.int32(N), np.int32(L), np.int32(base),
        block=(block_size, 1, 1), grid=(grid_size, 1))

    return result


mod = SourceModule("""
__global__ void fhtseq_inv_gpu(float *x, int N, int L)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = blockDim.x * gridDim.x;

    // Use shared memory to reduce global memory access and add padding if necessary
     __shared__ float sdata[32];

    for(int pos = tid; pos < N; pos += threadNum) {
        // Load data into shared memory
        sdata[threadIdx.x] = x[pos];
        __syncthreads();

        int k1 = N;
        int k2 = 1;
        int k3 = N >> 1;

        for(int i1 = 0; i1 < L; ++i1) {
            int L1 = 0;

            for(int i2 = 0; i2 < k2; ++i2) {
                // Precompute operations based on (i2 & 1) to avoid branching inside the loop
                bool isOdd = i2 & 1;
                for(int i3 = 0; i3 < k3; ++i3) {
                    int ii = i3 + L1;
                    int jj = ii + k3;

                    float temp1 = sdata[ii];
                    float temp2 = sdata[jj];

                    float add = temp1 + temp2;
                    float sub = temp1 - temp2;

                    // Use precomputed condition to avoid branching
                    sdata[ii] = isOdd ? sub : add;
                    sdata[jj] = isOdd ? add : sub;
                }
                L1 += k1;
            }
            k1 >>= 1;
            k2 <<= 1;
            k3 >>= 1;
        }

        // Write back to global memory with reduced synchronization
        x[pos] = sdata[threadIdx.x] / N;
        __syncthreads(); // Synchronize after writing to shared memory to ensure all threads have written their values
    }
}
""")

fhtseq_inv_gpu = mod.get_function("fhtseq_inv_gpu")

def fhtseq_inv_2(data, stream=None):
    N = len(data)
    L = int(np.log2(N))
    if ((L - np.floor(L)) > 0.0):
        raise ValueError("Length must be power of 2")

    block_dim = (256, 1, 1)
    grid_dim = (N + block_dim[0] - 1) // block_dim[0]

    x_gpu = gpuarray.to_gpu_async(bitrevorder_cuda(data).astype(np.float32), stream=stream)
    fhtseq_inv_gpu(x_gpu, np.int32(N),np.int32(L), block=block_dim, grid=(grid_dim, 1), stream=stream)
    result = x_gpu.get_async(stream=stream)
    return result


def fwht2d_GPU(xx):
    N = len(xx)
    xx1 = np.zeros((N, N), dtype=np.float32)
    xx_out = np.zeros((N, N), dtype=np.float32)

    # Create a stream for each direction of the 2D transform
    stream1 = cuda.Stream()
    stream2 = cuda.Stream()

    for i in range(N):
        xx1[i, :] = fhtseq_inv_2(xx[i, :], stream=stream1)

    stream1.synchronize()  # Ensure that the first direction is fully computed

    for j in range(N):
        xx_out[:, j] = fhtseq_inv_2(xx1[:, j], stream=stream2)

    stream2.synchronize()  # Ensure that the second direction is fully computed

    return xx_out



def getHSPIReconstruction_GPU( dataMat, nStep ):
    if (nStep == 2):
        spec = dataMat[:,:,0] - dataMat[:,:,1]
        img  = fwht2d_GPU(spec)
    return img, spec