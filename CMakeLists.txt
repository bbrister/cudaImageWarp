
include (ExternalProject)

cmake_minimum_required (VERSION 2.8)

project (cudaImageWarp)

# Paths 
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
list (APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/sift3d)

# Output directories
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Build flags
set (DEBUG_FLAGS "-g -ggdb3")
set (RELEASE_FLAGS "-O3 -DNDEBUG")

set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${RELEASE_FLAGS}")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")

# OS-specific build flags
if (APPLE)
        set(CMAKE_MACOSX_RPATH ON)
endif()

# Find CUDA
find_package (CUDA REQUIRED)

# Compile the library
cuda_add_library(cudaImageWarp SHARED cudaImageWarp.cu)
set_property(TARGET cudaImageWarp
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories (cudaImageWarp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Install the library
install (TARGETS cudaImageWarp
                RUNTIME DESTINATION lib 
                LIBRARY DESTINATION lib
                ARCHIVE DESTINATION bin
)

# Install the header file
set (CUDA_IMAGE_WARP_H_FILE cudaImageWarp.h)
install (FILES ${CUDA_IMAGE_WARP_H} DESTINATION include)

if (APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET cudaImageWarp
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif ()

add_subdirectory (test)
