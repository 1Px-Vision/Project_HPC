// digitrevorder_kernel.cu

extern "C" __global__ void digitrevorder_kernel(int *x, int *vec, int *result, int N, int L, int base) {
    __shared__ int s_vec[32];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        s_vec[i] = vec[i];
    }
    __syncthreads();

  
    if (idx < N) {
        int res = 0;
        int temp = x[idx];
        for (int k = L - 1; k >= 0; --k) {
            int div = temp / s_vec[k];
            res += div * s_vec[L - 1 - k];
            temp -= div * s_vec[k];
        }
        result[idx] = res;
    }
    
}
