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


def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max

def getHSPIReconstruction( dataMat, nStep ):
    if (nStep == 2):
        spec = dataMat[:,:,0] - dataMat[:,:,1]
        img  = fwht2d(spec)

    return img, spec


def fwht2d(xx):
   N = len(xx)
   xx1=np.zeros((N,N))
   for i in range(N):
        xx1[i,:] = fhtseq_inv(xx[i,:])


   xx =np.zeros((N,N))
   for j in range(N):
     xx[:,j] = fhtseq_inv(xx1[:,j].T).T


   return xx

def fhtseq_inv(data):
    N = len(data)
    L = np.log2(N)
    if ((L-np.floor(L)) > 0.0):
        raise (ValueError, "Length must be power of 2")
    x=bitrevorder(data)

    k1=N
    k2=1
    k3=int(N/2)
    for i1 in range(1,int(L+1)):  #Iteration stage
        L1=1
        for i2 in range(1,int(k2+1)):
            for i3 in range(1,int(k3+1)):
                ii=int(i3+L1-1)
                jj=int(ii+k3)
                temp1= x[ii-1]
                temp2 = x[jj-1]
                if (i2 % 2) == 0:
                    x[ii-1] = temp1 - temp2
                    x[jj-1] = temp1 + temp2

                else:
                    x[ii-1] = temp1 + temp2
                    x[jj-1] = temp1 - temp2

            L1=L1+k1
        k1 = round(k1/2)
        k2 = k2*2
        k3 = round(k3/2)

    return (1/N)*x

def bitrevorder(x):
    temp_x=np.arange(0,len(x))
    temp_y=digitrevorder(temp_x,2)
    return x[temp_y]


def digitrevorder(x,base):
    x = np.asarray(x)
    rem = N = len(x)
    L = 0
    while 1:
        if rem < base:
            break
        intd = rem // base
        if base*intd != rem:
            raise (ValueError, "Length of data must be power of base.")
        rem = intd
        L += 1
    vec = r_[[base**n for n in range(L)]]
    newx = x[np.newaxis,:]*vec[:,np.newaxis]
    # compute digits
    for k in range(L-1,-1,-1):
        newx[k] = x // vec[k]
        x = x - newx[k]*vec[k]
    # reverse digits
    newx = newx[::-1,:]
    x = 0*x
    # construct new result from reversed digts
    for k in range(L):
        x += newx[k]*vec[k]
    return x
