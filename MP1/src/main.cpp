#include "kernels.cuh"

int main(int argc, char** argv) {
    int NO_THREADS_PER_BLOCK = 256;
    // Uncomment this line to print device details
    // print_device_details();

    // Ensure the user passes the data directory as an argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_data_dir>\n";
        std::cerr << "Example: " << argv[0] << " data/0\n";
        return 1;
    }

    std::string data_dir = argv[1];

    // Loading the data into host memory
    std::vector<float> input0, input1, output_expected;
    if (!load_vector(data_dir + "/input0.raw", input0) ||
        !load_vector(data_dir + "/input1.raw", input1) ||
        !load_vector(data_dir + "/output.raw", output_expected)) {
        return 1; // Exit if files are missing
    }

    // Allocating host memory to hold GPU output
    int numElements = input0.size();
    std::vector<float> output_gpu(numElements, 0.0f);
    std::vector<float> output_cpu(numElements, 0.0f);

    std::cout << "Loaded datasets " << data_dir << "/... "<< "of size: "<< numElements << "\n";
    
    run_kernel(input0.data(), input1.data(), output_gpu.data(), numElements, NO_THREADS_PER_BLOCK);
    run_cpu(input0.data(), input1.data(), output_cpu.data(), numElements);

    verfiy_1d_results(output_gpu, output_expected, numElements);
    
    return 0;
}