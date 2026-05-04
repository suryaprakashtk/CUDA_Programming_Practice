#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include "kernels.cuh"

// Helper function to load 1D dataset
bool load_vector(const std::string& filepath, std::vector<float>& data) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << "\n";
        return false;
    }

    int length;
	// The number of elements
    file >> length;

    data.resize(length);
    for (int i = 0; i < length; ++i) {
        file >> data[i];
    }

    return true;
}

// Used to verify 1d results
bool verfiy_1d_results(std::vector<float>& expected, std::vector<float>& calculated, int numElements)
{
	std::cout << "Verifying results...\n";
	bool success = true;
    float max_error = 0.0f;
	// Floating point tolerance
    const float EPSILON = 1e-4;

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
        std::cout << "SUCCESS! All values match expected output. (Max Error: " << max_error << ")\n";
    } else {
        std::cout << "FAILED! Output does not match expected dataset.\n";
    }

	return success;
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