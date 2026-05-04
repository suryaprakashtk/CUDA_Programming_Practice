#include <iostream>
#include <chrono>
#include "kernels.cuh"

__global__ void hello_cuda(const float* in1, const float* in2, float* out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent segfaults if threads > len
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

void run_kernel(const float* input_1, const float* input_2, float* output, int numElements) {
    // Create threads and blocks
    int THREADS = 256;
    int BLOCKS = 1;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, 1, 1);
    dim3 blocks(BLOCKS, 1, 1);

    // Set size of input data
    size_t size = numElements * sizeof(float);
    float *device_input_1 = nullptr, *device_input_2 = nullptr, *device_output = nullptr;

    /*
        Allocate device memory using cudaMalloc()
        First argument is a ptr to a ptr which stores the address of device memory allocated
    */
    cudaMalloc((void**)&device_input_1, size);
    cudaMalloc((void**)&device_input_2, size);
    cudaMalloc((void**)&device_output, size);


    // 2. Copy Host Data to Device
    cudaMemcpy(device_input_1, input_1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_2, input_2, size, cudaMemcpyHostToDevice);

    // Starting time to measure kernel execution time.
    auto start = std::chrono::high_resolution_clock::now();
    // Luanch kernel
    hello_cuda<<<blocks, threads>>>(device_input_1, device_input_2, device_output, numElements);

    // cudaDeviceSynchronize waits until all kernel code is done
    cudaDeviceSynchronize();

    // Stopping time to measure kernel execution time.
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< " ms\n";

    // Copy data to the host
    cudaMemcpy(output, device_output, size, cudaMemcpyDeviceToHost);

    // 6. Free Device Memory
    cudaFree(device_input_1);
    cudaFree(device_input_2);
    cudaFree(device_output);

    return;
}
