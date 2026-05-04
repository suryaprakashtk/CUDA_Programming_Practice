#include <iostream>
#include "kernels.cuh"
#include <chrono>

__global__ void hello_cuda() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

void run_kernel() {
    // Set size of input data
    size_t bytes = 1020 * sizeof(int);

    // Allocate and initialize memory in host


    /*
        Allocate device memory using cudaMalloc()
        First argument is a ptr to a ptr which stores the address of device memory allocated
    */
    int *pointer_of_pointer_to_device_memory;
    cudaMalloc(&pointer_of_pointer_to_device_memory, bytes);


    // Copy data to the device
    // cudaMemcpy(dst_mem, src_mem, bytes, cudaMemcpyHostToDevice);


    // Create threads and blocks
    int THREADS = 32;
    int BLOCKS = 2;
    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, 1, 1);
    dim3 blocks(BLOCKS, 1, 1);

    // Starting time to measure kernel execution time.
    auto start = std::chrono::high_resolution_clock::now();
    // Luanch kernel
    hello_cuda<<<blocks, threads>>>();

    // cudaDeviceSynchronize waits until all kernel code is done
    cudaDeviceSynchronize();

    // Stopping time to measure kernel execution time.
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< " ms\n";

    // Copy data to the host
    // cudaMemcpy(dst_mem, src_mem, bytes, cudaMemcpyDeviceToHost);

    // Validate GPU and CPU results
    // verify_result();

    std::cout << "DONE\n";
}