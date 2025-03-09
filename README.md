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

## Evaluation Algorithms 

Table 1. Evaluating the simulation with Corel 1K dataset applying reconstruction of NIR-SPI Images using various reconstruction strategies, including TVAL3 (CPU 30 % samples), AP (CPU 60 % samples), OMP (CPU 30 % samples ) and FHSI

| Method  | PSNR (dB) |SSIM |
| ------------- | ------------- |------------- |
| FHSI (CPU)  | 21.7  |0.7  |
| FHSI (GPU)_pre  | 24.38  |0.71  |
| OMP  | 15.28  |0.56  |
|AP  | 14.87  |0.42  |
| TVLA3  | 22.40  |0.68 |

Table 2. Evaluating laboratory reconstruction of NIR-SPI Images at a distance of 60 cm using various reconstruction strategies including TVAL3, AP, OMP, FHSI (CPU), and FHSI (GPU)_pre.

| Method  | Memory usage % |Execution time (ms) |SpeedUp % |
| ------------- | ------------- |------------- |------------- |
| FHSI (CPU)  | 1.8 |340 |--- |
| FHSI (GPU)  | 1.9  |45  |x7.5|
| FHSI (CPU)_pre  | 1.8 |43 |x7.9 |
| FHSI (GPU)_pre  | 1.5 |20-34  |x10|

Table 3. Evaluating various SPI reconstruction algorithms OMP (30% samples), TVAL3 (30% samples), AP (60% samples), and FHSI): a comparative analysis of memory consumption, execution time (ms), and speedup performance between OMP on CPU and other algorithms for 64x64 images. 

| Method  | Memory usage % |Execution time (ms) |SpeedUp % |
| ------------- | ------------- |------------- |------------- |
| OMP  | 10 |250-400 |--- |
| TVAL3  | 15  |>1000  |---|
|AP  | >20 |>5000 |--- |
|OMP (GPU)  | <4  |50-70 |x3.5 |
| FHSI (GPU)_pre  | 1.5 |20-34  |x10|

# Cypthon module

## Setup the Project Structure
````
project/
├── cython_CUDA_FHSPI.pyx     # Cython code integrating CUDA for FFT and Hadamard transforms 
├── digitrevorder_kernel.cu    # CUDA kernel for digit-reversed order computation
├── digitrevorder_kernel.cuh   # Header file for digitrevorder kernel
├── fhtseq_inv_gpu_kernel.cu   # CUDA kernel for inverse fast Hadamard transform
├── fhtseq_inv_gpu_kernel.cuh  # Header file for fhtseq_inv_gpu kernel
├── setup.py                   # Setup script to compile Cython and CUDA files
└── main.py                    # Main script demonstrating usage with .mat files
````

## Compile your Cython module

````
python setup.py build_ext --inplace
````

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes.

## Contact
For any questions or inquiries, please contact caoq@1px-vision.com
