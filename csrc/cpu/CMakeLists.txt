# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

cmake_minimum_required(VERSION 3.18)
project(cpu_sparse_attention)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AVX-512 flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512vl -mavx512bw")

# Create the extension
add_library(cpu_sparse_attention SHARED
    sparse_attention.cpp
)

target_include_directories(cpu_sparse_attention PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cpu_sparse_attention
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Set output name
set_target_properties(cpu_sparse_attention PROPERTIES
    PREFIX ""
    OUTPUT_NAME "cpu_sparse_attention"
)
