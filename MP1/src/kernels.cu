
#include "kernels.cuh"

__global__ void hello_cuda(const float* in1, const float* in2, float* out, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent segfaults if threads > len
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

void run_kernel(const float* input_1, const float* input_2, float* output, int numElements, int threadsPerBlock = 256) {
    // Create threads and block
    int noOfBlocks = ceil(numElements/(float) threadsPerBlock);

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(threadsPerBlock, 1, 1);
    dim3 blocks(noOfBlocks, 1, 1);

    // Set size of input data in terms of bytes
    size_t size = numElements * sizeof(float);
    float *device_input_1 = nullptr, *device_input_2 = nullptr, *device_output = nullptr;

    // Starting time to measure kernel execution time.
    auto start = std::chrono::high_resolution_clock::now();

    /*
        Allocate device memory using cudaMalloc()
        First argument is a ptr to a ptr which stores the address of device memory allocated
        Reason: We are providing a pointer that will end up pointing to the allocated device memory.
        The allocator needs to change the value and so we need to provide address of THAT pointer and not the pointer itself.
    */
    cudaMalloc((void**)&device_input_1, size);
    cudaMalloc((void**)&device_input_2, size);
    cudaMalloc((void**)&device_output, size);


    // 2. Copy Host Data to Device
    cudaMemcpy(device_input_1, input_1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_2, input_2, size, cudaMemcpyHostToDevice);

    
    // Luanch kernel
    hello_cuda<<<blocks, threads>>>(device_input_1, device_input_2, device_output, numElements);

    // cudaDeviceSynchronize waits until all kernel code is done
    cudaDeviceSynchronize();

    // Copy data to the host
    cudaMemcpy(output, device_output, size, cudaMemcpyDeviceToHost);

    // 6. Free Device Memory
    cudaFree(device_input_1);
    cudaFree(device_input_2);
    cudaFree(device_output);

    // Stopping time to measure kernel execution time.
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< " ms\n";

    return;
}
