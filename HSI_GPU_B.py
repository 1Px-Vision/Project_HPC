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

mod = SourceModule("""
__global__ void digitrevorder_kernel(int *x, int *vec, int *result, int N, int L, int base) {
    __shared__ int s_vec[32];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIdx.x < L) {
        s_vec[threadIdx.x] = vec[threadIdx.x];
    }
    __syncthreads();  // Ensure all the required vec values are loaded into shared memory

    if (idx < N) {
        int temp = x[idx];
        int res = 0;
        for (int k = L - 1; k >= 0; --k) {
            int current_digit = temp / s_vec[k];
            res += current_digit * s_vec[L - 1 - k];
            temp -= current_digit * s_vec[k];
        }
        result[idx] = res;
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
__global__ void fhtseq_inv_gpu(float *x, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= N) return;

    int L = log2f(N);

    int k1 = N;
    int k2 = 1;
    int k3 = N / 2;

    for(int i1 = 1; i1 <= L; i1++)
    {
        int L1 = 1;
        for(int i2 = 1; i2 <= k2; i2++)
        {
            for(int i3 = 1; i3 <= k3; i3++)
            {
                int ii = i3 + L1 - 1;
                int jj = ii + k3;

                float temp1 = x[ii-1];
                float temp2 = x[jj-1];

                if(i2 % 2 == 0)
                {
                    x[ii-1] = temp1 - temp2;
                    x[jj-1] = temp1 + temp2;
                }
                else
                {
                    x[ii-1] = temp1 + temp2;
                    x[jj-1] = temp1 - temp2;
                }
            }
            L1 = L1 + k1;
        }
        k1 /= 2;
        k2 *= 2;
        k3 /= 2;
    }

    x[tid] = x[tid] / N;
}
""")

fhtseq_inv_gpu = mod.get_function("fhtseq_inv_gpu")

def fhtseq_inv_2(data):
    N = len(data)
    L = np.log2(N)
    if ((L - np.floor(L)) > 0.0):
        raise ValueError("Length must be power of 2")

    block_dim = (256, 1, 1)
    grid_dim = (N + block_dim[0] - 1) // block_dim[0]

    x_gpu = gpuarray.to_gpu(bitrevorder_cuda(data).astype(np.float32))
    fhtseq_inv_gpu(x_gpu, np.int32(N), block=block_dim, grid=(grid_dim, 1))
    return x_gpu.get()


def fwht2d_GPU(xx):
    N = len(xx)
    xx1 = np.zeros((N, N))
    for i in range(N):
        xx1[i, :] = fhtseq_inv_2(xx[i, :])

    xx_out = np.zeros((N, N))
    for j in range(N):
        xx_out[:, j] = fhtseq_inv_2(xx1[:, j])

    return xx_out


def getHSPIReconstruction_GPU( dataMat, nStep ):
    if (nStep == 2):
        spec = dataMat[:,:,0] - dataMat[:,:,1]
        img  = fwht2d_GPU(spec)
    return img, spec