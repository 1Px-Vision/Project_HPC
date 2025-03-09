# main.py
import os
import numpy as np
import scipy.io
from cython_cuda_fft import getHSPIReconstruction_GPU

if __name__ == '__main__':
    param_path = '/content/parameter/parameter_dt_5.mat'
    data_path = '/content/data/dat_dt_5.mat'

    if not os.path.exists(param_path) or not os.path.exists(data_path):
        print('Parameter or data files not found. Please update the paths.')
        exit(0)

    m_parameter = scipy.io.loadmat(param_path)
    m_image = scipy.io.loadmat(data_path)

    nStep = 2
    nCoeft = 64 * 64
    iRow = m_parameter['iRow1'][0]
    jCol = m_parameter['jCol1'][0]

    intensity_mat = np.zeros((64, 64, nStep), dtype=np.float32)
    cont1 = 0
    for ii in range(nCoeft):
        for jj in range(nStep):
            intensity_mat[iRow[ii]-1, jCol[ii]-1, jj] = m_image['measurement'][0, cont1]
            cont1 += 1

    # Call your Cython-compiled CUDA reconstruction function
    img_spi_GPU_p, spec = getHSPIReconstruction_GPU(intensity_mat, nStep)

    # Normalize the output
    img_spi_GPU_p_min = img_spi_GPU_p.min()
    img_spi_GPU_p_max = img_spi_GPU_p.max()

    if img_spi_GPU_p_max - img_spi_GPU_p_min > 1e-8:
        img_spi_GPU_p = (img_spi_GPU_p - img_spi_GPU_p_min) / (img_spi_GPU_p_max - img_spi_GPU_p_min)
    else:
        img_spi_GPU_p = np.zeros_like(img_spi_GPU_p)

    print("Reconstruction completed successfully.")
