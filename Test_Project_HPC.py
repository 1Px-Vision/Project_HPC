from HSI_CPU_B import getHSPIReconstruction
from HSI_CPU_P import getHSPIReconstruction_P
from HSI_GPU_B import getHSPIReconstruction_GPU
from HSI_GPU_P import getHSPIReconstruction_GPU_P

import scipy.io
from time import time
import numpy as np
from time import time
from matplotlib import pyplot as plt

if __name__=="__main__":
    m_parameter=scipy.io.loadmat('.../parameter_hsi.mat')
    m_image=scipy.io.loadmat('.../dat_HSI_64.mat')


    nStep=2
    nCoeft=64*64
    iRow=m_parameter['iRow1'][0]
    jCol=m_parameter['jCol1'][0]


    intensity_mat=np.zeros((64,64,nStep))
    IntensityMat=np.zeros((64,64,nStep))

    cont1=0
    for ii in range(nCoeft):
        for jj in range(nStep):
            intensity_mat[iRow[ii]-1,jCol[ii]-1,jj]=m_image['measurement'][0,cont1]#img_dat[cont1]/(32)
            cont1=cont1+1


    %timeit img_spi, spec=getHSPIReconstruction(intensity_mat, nStep )#CPU
    %timeit img_spi_p, spec=getHSPIReconstruction_P(intensity_mat, nStep )#CPU Pre-processed

    %timeit img_spi_GPU, spec=getHSPIReconstruction_GPU(intensity_mat, nStep )#GPU Pre-processed
    %timeit img_spi_GPU_p, spec=getHSPIReconstruction_GPU_P(intensity_mat, nStep )#GPU Pre-processed


    plt.imshow(img_spi,cmap='gray');
    plt.title("CPU-OMP")