# Project_HPC

# Abstract

The recent advancements in edge computing power are primarily attributable to technological innovations enabling accelerators with extensive hardware parallelism. One practical application is in computer imaging (CI), where GPU acceleration is pivotal, especially in reconstructing 2D images through techniques like Single-Pixel Imaging (SPI). In SPI, compressive sensing (CS) algorithms, deep learning, and Fourier transformation are essential for 2D image reconstruction. These algorithms benefit significantly from parallelism, enhancing performance by optimizing processing times. Strategies such as optimizing memory access, unrolling loops, crafting kernels to minimize operation counts, utilizing asynchronous operations, and maximizing the number of active threads and warps are employed to fully exploit the GPU’s capabilities. Integrating embedded GPUs becomes essential for algorithmic optimization on SoC-GPUs in various lab scenarios. This study swiftly enhances the Fast Harley Transform (FHT) for 2D image reconstruction on Nvidia’s Xavier platform. By applying diverse parallelism strategies in PyCUDA, we achieve an approximate acceleration x 10, bringing processing times close to real-time.

![HPC](https://github.com/1Px-Vision/Project_HPC/assets/150855410/13bd4b32-6cbf-4291-bde1-ae11de2b72e6)

## File Code FHSI
* Data configuration HSI parameter_hsi.mat
* Data measuring dat_HSI_64.mat

* File FAST HARTLEY TRANSFORM (FHT) Single Pixel Imaging (SPI)
* File HSI_CPU_B-> FHSI CPU  
* File HSI_CPU_P-> FHSI CPU with Pre-processed

* File HSI_GPU_B-> FHSI GPU Kernel PyCUDA  
* File HSI_GPU_P-> FHSI GPU Kernel PyCUDA with Pre-processed    

Table 1. Evaluating the simulation with Corel 1K dataset applying reconstruction of NIR-SPI Images using various reconstruction strategies, including TVAL3 (CPU 30 % samples), AP (CPU 60 % samples), OMP (CPU 30 % samples ) and FHSI

| Methods  | PSNR (dB) |SSIM |
| ------------- | ------------- |------------- |
| FHSI (CPU)  | 21.7  |0.7  |
| FHSI (GPU)_pre  | 24.38  |0.71  |
| OMP  | 15.28  |0.56  |
|AP  | 14.87  |0.42  |
| TVLA3  | 22.40  |0.68 |
