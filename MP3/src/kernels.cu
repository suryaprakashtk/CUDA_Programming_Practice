#include "kernels.cuh"

__global__ void cuda_mat_mul_basic(const float* mat0, const float* mat1, float* out, int mat0_row, int mat0_col, int mat1_row, int mat1_col) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int p_index = row * mat1_col + col;
    float sum = 0;

    // Boundary check
    if (row < mat0_row && col < mat1_col) 
    {
        for(int k = 0 ; k < mat0_col; k++){
            int mat0_index = row * mat0_col + k;
            int mat1_index = k * mat1_col + col;
            if(mat0_index < mat0_row * mat0_col && mat1_index < mat1_row * mat1_col)
                sum = sum + mat0[mat0_index] * mat1[mat1_index];
        }
        out[p_index] = sum;
    }
    return;
}

__global__ void cuda_mat_mul_tiled_coaleased(const float* mat0, const float* mat1, float* result, int mat0_row, int mat0_col, int mat1_row, int mat1_col) {

    __shared__ float sub_mat0[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sub_mat1[TILE_WIDTH][TILE_WIDTH];

    int result_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int result_col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int result_index = result_row * mat1_col + result_col;
    
    int no_of_faces = ceil(mat0_col/(float) TILE_WIDTH); // Same number of faces as mat0_col = mat1_row

    float result_value = 0;
    for(int face = 0; face < no_of_faces; face++){
        int mat0_index = result_row * mat0_col + face * TILE_WIDTH + threadIdx.x;
        int mat1_index = (face * TILE_WIDTH + threadIdx.y) * mat1_col + result_col;
        if(result_row < mat0_row && (face * TILE_WIDTH + threadIdx.x) < mat0_col)
            sub_mat0[threadIdx.y][threadIdx.x] = mat0[mat0_index];
        else
            sub_mat0[threadIdx.y][threadIdx.x] = 0.0f;
        if((face * TILE_WIDTH + threadIdx.y) < mat1_row && result_col < mat1_col)
            sub_mat1[threadIdx.y][threadIdx.x] = mat1[mat1_index];
        else
            sub_mat1[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        for(int k = 0 ; k < TILE_WIDTH; k++){
            result_value = result_value + sub_mat0[threadIdx.y][k] * sub_mat1[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (result_row < mat0_row && result_col < mat1_col){
        result[result_index] = result_value;
    }
    return;
}

void run_kernel(const float* input_1, const float* input_2, float* output, std::vector<int>& input0_dim, std::vector<int>& input1_dim) {
    // Create threads and block
    int noOfBlocksX = ceil(input1_dim[1]/(float) TILE_WIDTH); // P Matrix column from Matrix 2
    int noOfBlocksY = ceil(input0_dim[0]/(float) TILE_WIDTH); // P Matrix row from Matrix 1

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(TILE_WIDTH, TILE_WIDTH, 1);
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

    
    // Launch kernel
    cuda_mat_mul_tiled_coaleased<<<blocks, threads>>>(device_input_1, device_input_2, device_output, input0_dim[0], input0_dim[1], input1_dim[0], input1_dim[1]);

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
