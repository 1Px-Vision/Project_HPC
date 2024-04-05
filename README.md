# Project_HPC

# Data configuration HSI parameter_hsi.mat
# Data measuring dat_HSI_64.mat

# File FAST HARTLEY TRANSFORM (FHT) Single Pixel Imaging (SPI)
# File HSI_CPU_B-> FHT CPU  
# File HSI_CPU_P-> FHT CPU with Pre-processed

# File HSI_GPU_B-> FHT GPU Kernel PyCUDA  
# File HSI_GPU_P-> FHT GPU Kernel PyCUDA with Pre-processed    


# Abstract

The recent advancements in edge computing power are largely attributable to technological innovations that enable accelerators with extensive hardware parallelism. One practical application is in computer imaging (CI), where GPU acceleration is pivotal, especially in reconstructing 2D images through techniques like Single-Pixel Imaging (SPI). In SPI, compressive sensing (CS) algorithms, deep learning, and Fourier transformation are essential for 2D image reconstruction. These algorithms benefit significantly from parallelism, enhancing performance by optimizing processing times. Strategies such as optimizing memory access, unrolling loops, crafting kernels to minimize operation counts, utilizing asynchronous operations, and maximizing the number of active threads and warps are employed to fully exploit the GPU’s capabilities. In various lab scenarios, integrating embedded GPUs becomes essential for algorithmic optimization on SoC-GPUs. This study concentrates on swiftly enhancing the Fast Harley Transform (FHT) for 2D image reconstruction on Nvidia’s Xavier platform. By applying diverse parallelism strategies in PyCUDA, we achieve an approximate acceleration x 10, bringing processing times close to real-time
