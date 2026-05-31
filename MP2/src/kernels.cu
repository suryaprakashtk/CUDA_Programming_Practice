
#include "kernels.cuh"

__global__ void cuda_mat_mul_basic(const float* in0, const float* in1, float* out, int in0_row, int in0_col, int in1_row, int in1_col) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int curr_index = i * in1_col + j;
    float sum = 0;
    // Boundary check
    if (i < in0_row && j < in1_col) 
    {
        for(int k = 0 ; k < in0_col; k++){
            int mat0_index = i * in0_col + k;
            int mat1_index = k * in1_col + j;
            if(mat0_index < in0_row * in0_col && mat1_index < in1_row * in1_col)
                sum = sum + in0[mat0_index] * in1[mat1_index];
        }
        out[curr_index] = sum;
    }
    return;
}

void run_kernel(const float* input_1, const float* input_2, float* output, std::vector<int>& input0_dim, std::vector<int>& input1_dim, int threadsPerBlockX = 16, int threadsPerBlockY = 16) {
    // Create threads and block
    int noOfBlocksX = ceil(input0_dim[0]/(float) threadsPerBlockX);
    int noOfBlocksY = ceil(input1_dim[1]/(float) threadsPerBlockY);

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(threadsPerBlockX, threadsPerBlockY, 1);
    dim3 blocks(noOfBlocksX, noOfBlocksY, 1);

    float *device_input_1 = nullptr, *device_input_2 = nullptr, *device_output = nullptr;

    // Starting time to measure kernel execution time.
    auto start = std::chrono::high_resolution_clock::now();

    /*
        Allocate device memory using cudaMalloc()
        First argument is a ptr to a ptr which stores the address of device memory allocated
        Reason: We are providing a pointer that will end up pointing to the allocated device memory.
        The allocator needs to change the value and so we need to provide address of THAT pointer and not the pointer itself.
    */
    cudaMalloc((void**)&device_input_1, input0_dim[0] * input0_dim[1] * sizeof(float));
    cudaMalloc((void**)&device_input_2, input1_dim[0] * input1_dim[1] * sizeof(float));
    cudaMalloc((void**)&device_output, input0_dim[0] * input1_dim[1] * sizeof(float));


    // 2. Copy Host Data to Device
    cudaMemcpy(device_input_1, input_1, input0_dim[0] * input0_dim[1] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input_2, input_2, input1_dim[0] * input1_dim[1] * sizeof(float), cudaMemcpyHostToDevice);

    
    // Luanch kernel
    cuda_mat_mul_basic<<<blocks, threads>>>(device_input_1, device_input_2, device_output, input0_dim[0], input0_dim[1], input1_dim[0], input1_dim[1]);

    // cudaDeviceSynchronize waits until all kernel code is done
    cudaDeviceSynchronize();

    // Copy data to the host
    cudaMemcpy(output, device_output, input0_dim[0] * input1_dim[1] * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Free Device Memory
    cudaFree(device_input_1);
    cudaFree(device_input_2);
    cudaFree(device_output);

    // Stopping time to measure kernel execution time.
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< " ms\n";

    return;
}
