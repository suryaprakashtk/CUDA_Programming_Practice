/*
Starter code for future practice sessions for CUDA Programs
Author: Surya Prakash
Commands
nvcc -o out start_code.cu
./out
*/

#include <iostream>
#include <chrono>


void print_device_details();

// __global__ means this is called from the CPU, and runs on the GPU
__global__ void yourKernel()
{
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Addressing (row, col) (%d, %d)\n", row, col);
}


int main(int argc, char**argv) {
    print_device_details(); //uncomment this line to print device details
    // cudaDeviceSynchronize is to put a blocking step in Host until previous kernels are completed
    cudaDeviceSynchronize();

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
    yourKernel<<<blocks, threads>>>();

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
    return 0;
}


// Prints all the information of device 0 connected to the host
void print_device_details()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("===== CUDA DEVICE PROPERTIES =====\n\n");

    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    printf("\n--- Hardware Limits ---\n");
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("MultiProcessor count: %d\n", prop.multiProcessorCount);

    printf("\n--- Memory ---\n");
    printf("Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Shared memory per multiprocessor: %.2f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Registers per multiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("L2 cache size: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));

    printf("\n--- Execution ---\n");
    printf("Warp size: %d\n", prop.warpSize);
    printf("Clock rate: %d kHz\n", prop.clockRate);
    printf("Memory clock rate: %d kHz\n", prop.memoryClockRate);

    printf("\n--- Features ---\n");
    printf("Concurrent kernels: %d\n", prop.concurrentKernels);
    printf("Device overlap: %d\n", prop.deviceOverlap);
    printf("Unified addressing: %d\n", prop.unifiedAddressing);
    printf("Managed memory: %d\n", prop.managedMemory);

    printf("\n--- Limits ---\n");
    printf("Max grid size: (%d, %d, %d)\n",
           prop.maxGridSize[0],
           prop.maxGridSize[1],
           prop.maxGridSize[2]);

    printf("Max block size: (%d, %d, %d)\n",
           prop.maxThreadsDim[0],
           prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);

    printf("\n=================================\n");

    return;
}