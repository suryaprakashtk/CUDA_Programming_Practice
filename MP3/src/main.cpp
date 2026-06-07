#include "kernels.cuh"

int main(int argc, char** argv) {
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
    std::vector<int> input0_dim, input1_dim, output_expected_dim;
    if (!load_vector(data_dir + "/input0.raw", input0, input0_dim) ||
        !load_vector(data_dir + "/input1.raw", input1, input1_dim) ||
        !load_vector(data_dir + "/output.raw", output_expected, output_expected_dim)) {
        return 1; // Exit if files are missing
    }

    // Allocating host memory to hold GPU output
    std::vector<float> output_gpu(output_expected_dim[0] * output_expected_dim[1], 0.0f);
    std::vector<float> output_cpu(output_expected_dim[0] * output_expected_dim[1], 0.0f);

    std::cout << "Loaded datasets " << data_dir << "\n";
    std::cout << "Input 0 has dimension " << input0_dim[0] << " , " << input0_dim[1]<< "\n";
    std::cout << "Input 1 has dimension " << input1_dim[0] << " , " << input1_dim[1]<< "\n";
    std::cout << "Expected Output dimension " << output_expected_dim[0] << " , " << output_expected_dim[1]<< "\n";
    
    run_kernel(input0.data(), input1.data(), output_gpu.data(), input0_dim, input1_dim);
    run_cpu(input0.data(), input1.data(), output_cpu.data(), input0_dim, input1_dim);

    verfiy_results(output_gpu, output_expected, output_expected_dim[0] * output_expected_dim[1]);
    
    return 0;
}