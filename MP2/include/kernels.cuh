#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

// Declare the host-callable wrapper function.
// main.cpp will include this to know that run_kernel() exists.
void run_kernel(const float* input_1, const float* input_2, float* output, std::vector<int>& input0_dim, std::vector<int>& input1_dim, int threadsPerBlockX, int threadsPerBlockY);
void run_cpu(const float* input_1, const float* input_2, float* output, std::vector<int>& input0_dim, std::vector<int>& input1_dim);
void print_device_details();
bool load_vector(const std::string& filepath, std::vector<float>& data, std::vector<int>& dimension);
bool verfiy_results(std::vector<float>& expected, std::vector<float>& calculated, int numElements);
