#include "kernels.cuh"

// Helper function to load 1D dataset
bool load_vector(const std::string& filepath, std::vector<float>& data, std::vector<int>& data_dim) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << "\n";
        return false;
    }

    int row, col;
	// The number of elements
    file >> row >> col;
    data_dim.push_back(row);
    data_dim.push_back(col);

    data.resize(row * col);
    for (int i = 0; i < row * col; ++i) {
        file >> data[i];
    }

    return true;
}

// Used to verify 1d results
bool verfiy_results(std::vector<float>& expected, std::vector<float>& calculated, int numElements)
{
	bool success = true;
    float max_error = 0.0f;
	// Floating point tolerance
    const float EPSILON = 1e-2;

	for (int i = 0; i < numElements; ++i) {
        float diff = std::abs(expected[i] - calculated[i]);
        if (diff > max_error) max_error = diff;

        if (diff > EPSILON) {
            std::cerr << "Mismatch at index " << i << ": Expected " 
                      << expected[i] << " but got " << calculated[i] << "\n";
            success = false;
			// Stop at first error to prevent terminal flooding
            break;
        }
    }

	if (success) {
        std::cout << "SUCCESS!\n";
    } else {
        std::cout << "FAILED! FAILED! FAILED! FAILED! FAILED! FAILED! FAILED!\n";
    }

	return success;
}

void run_cpu(const float* input_0, const float* input_1, float* output, std::vector<int>& input0_dim, std::vector<int>& input1_dim)
{
    auto start = std::chrono::high_resolution_clock::now();

    if(input0_dim[1] != input1_dim[0]){
        std::cout<< "Dimension mismacth matrix cannot be multiplies" << "\n";
        return;
    }
    for(int i = 0; i < input0_dim[0]; i++){
        for(int j = 0; j < input1_dim[1]; j++){
            
            int curr_index = i * input1_dim[1] + j;
            float sum = 0.0;
            for(int k = 0 ; k<input0_dim[1]; k++){
                int mat0_index = i * input0_dim[1] + k;
                int mat1_index = k * input1_dim[1] + j;
                sum = sum + input_0[mat0_index] * input_1[mat1_index];
            }
            output[curr_index] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< " ms\n";
    
    return;
}


// Printing GPU Device Parameters
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