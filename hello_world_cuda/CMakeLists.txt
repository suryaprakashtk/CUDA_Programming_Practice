cmake_minimum_required(VERSION 3.18)
project(CudaPractice LANGUAGES CXX CUDA)

# 1. Specify the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. Tell CMake to find the GPU architecture automatically
# "all" is safe, but "native" optimizes for your specific machine
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# 3. Add include directories so headers are found
include_directories(include)

# 1. Search the src/ directory for all .cpp and .cu files and store them in variables
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS "src/*.cu")

# 2. Pass those variables to your executable
add_executable(cuda_app 
    ${CPP_SOURCES} 
    ${CU_SOURCES}
)

# # ==============================================================================
# # OPTIONAL: CUDA Libraries (Uncomment when needed)
# # ==============================================================================

# # 1. Find the CUDA Toolkit libraries
# find_package(CUDAToolkit REQUIRED)

# # 2. Link the specific libraries you need to your executable (e.g., cuBLAS and cuRAND)
# target_link_libraries(cuda_app PRIVATE CUDA::cublas CUDA::curand)