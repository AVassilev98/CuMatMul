cmake_minimum_required(VERSION 3.10)
project(CuMatMul LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)

add_executable(CuMatMul
    main.cu)

target_link_libraries(CuMatMul PRIVATE CUDA::cublas CUDA::curand)