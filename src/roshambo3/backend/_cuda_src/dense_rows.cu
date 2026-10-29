// See dense_rows.cuh.
#include "dense_rows.cuh"
#include "device_guard.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

__global__ void dense_library_rows_kernel(int * __restrict__ out, int n_query,
                                          long rows_per_query){
    const long stride = (long)blockDim.x * gridDim.x;
    for (int q = blockIdx.y; q < n_query; q += gridDim.y){
        int * row = out + (size_t)q * rows_per_query;
        for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
             t < rows_per_query; t += stride)
            row[t] = (int)t;
    }
}

void dense_library_rows(int * d_out, int n_query, long rows_per_query,
                        int device_id){
    if (n_query <= 0 || rows_per_query <= 0) return;
    ScopedDevice on_device(device_id);
    const int TPB = 256;
    int nblk = (int)std::min<long>(65535, (rows_per_query + TPB - 1) / TPB);
    dim3 grid((unsigned)nblk, (unsigned)std::min(65535, n_query));
    dense_library_rows_kernel<<<grid, TPB>>>(d_out, n_query, rows_per_query);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("roshambo3.cuda: dense_library_rows: ")
                                 + cudaGetErrorString(e));
}
