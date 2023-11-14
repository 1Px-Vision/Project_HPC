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

def computeBitRevOrder(N):
    return np.array([int('{:0{width}b}'.format(i, width=int(np.log2(N)))[::-1], 2) for i in range(N)])


def getHSPIReconstruction_P(dataMat, nStep):
    if nStep == 2:
        spec = dataMat[:, :, 0] - dataMat[:, :, 1]
        img = fwht2d_P(spec)
        return img, spec
    else:
        raise ValueError("nStep must be 2")

def fwht2d_P(matrix):
    N = matrix.shape[0]
    if not np.log2(N).is_integer():
        raise ValueError("Matrix dimensions must be a power of 2")

    bit_rev_order = computeBitRevOrder(N)

    transformed_rows = np.array([fhtseq_inv_P(row, bit_rev_order) for row in matrix])
    transformed_cols = np.array([fhtseq_inv_P(col, bit_rev_order) for col in transformed_rows.T]).T

    return transformed_cols

def fhtseq_inv_P(data, bit_rev_order):
    N = len(data)
    x = data[bit_rev_order]

    k1 = N
    k2 = 1
    k3 = N // 2

    for i1 in range(1, int(np.log2(N)) + 1):
        L1 = 1
        for i2 in range(k2):
            for i3 in range(k3):
                ii = i3 + L1 - 1  # Adjusted to ensure ii is within bounds
                jj = ii + k3
                if jj >= N:  # Check to ensure jj does not exceed bounds
                    continue
                temp1 = x[ii]
                temp2 = x[jj]
                x[ii] = (temp1 + temp2) if (i2 % 2) == 0 else (temp1 - temp2)
                x[jj] = (temp1 - temp2) if (i2 % 2) == 0 else (temp1 + temp2)
            L1 += k1
        k1 //= 2
        k2 *= 2
        k3 //= 2

    return x / N
