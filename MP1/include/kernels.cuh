#pragma once
#include <vector>
#include <string>

// Declare the host-callable wrapper function.
// main.cpp will include this to know that run_kernel() exists.
void run_kernel(const float* input_1, const float* input_2, float* output, int numElements);
void print_device_details();
bool load_vector(const std::string& filepath, std::vector<float>& data);
bool verfiy_1d_results(std::vector<float>& expected, std::vector<float>& calculated, int numElements);
