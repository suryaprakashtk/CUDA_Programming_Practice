#pragma once

// Declare the host-callable wrapper function.
// main.cpp will include this to know that run_kernel() exists.
void run_kernel();
void print_device_details();
