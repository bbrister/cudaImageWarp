#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <sift3d/imtypes.h>

#include "cudaImageWarp.h" // Need this to get C linkage on exported functions

#define DIVC(x, y)  (((x) + (y) + 1) / (y)) // Divide integers and ceil
#define AFFINE_WARP(x, y, z, f4) /* Warp using a float4 */ \
        (x * f4.x + y * f4.y + z * f4.z + f4.w)

/********************/
/* CUDA ERROR CHECK */
/********************/
typedef unsigned int  uint;

texture<float, 3, cudaReadModeElementType> tex;  // 3D texture

/************************************************/
/* TEXTURE-BASED TRILINEAR INTERPOLATION KERNEL */
/************************************************/
__global__ void
warp(float *d_output, const uint nx, const uint ny, const uint nz, 
        const float4 xWarp, const float4 yWarp, const float4 zWarp)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check boundaries
    if (x >= nx || y >= ny || z >= nz)
        return;

    const uint y_stride = nx;
    const uint z_stride = nx * ny;

    // Read from the 3D texture
    const float xs = AFFINE_WARP(x, y, z, xWarp);
    const float ys = AFFINE_WARP(x, y, z, yWarp);
    const float zs = AFFINE_WARP(x, y, z, zWarp);
    const float voxel = tex3D(tex, xs, ys, zs);

    // Write the output
    const uint idx = z * z_stride + y * y_stride + x;
    d_output[idx] = voxel;
}

/* Warp an image in-place.  Params is an array of 12 floats, in row-major 
 * order. */
int cuda_image_warp(Image *const im, const float *const params) {

    // Convert the input
    const float4 xWarp = {params[0], params[1], params[2], params[3]};
    const float4 yWarp = {params[4], params[5], params[6], params[7]};
    const float4 zWarp = {params[8], params[9], params[10], params[11]}; 

    // Intermediates
    float *d_output = NULL;

#define CLEANUP { \
    if (d_output != NULL) \
        cudaFree(d_output); \
} 

#define gpuAssert(code, file, line) { \
    if (code != cudaSuccess) { \
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), \
                file, line); \
        CLEANUP \
        return -1; \
       }  \
} \

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

    // --- Allocate device memory for output
    const size_t im_mem_size = im->size * sizeof(float);
    gpuErrchk(cudaMalloc((void**)&d_output, im_mem_size));

    // --- Create 3D array
    const cudaExtent volumeSize = make_cudaExtent(im->nx, im->ny, im->nz);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *d_inputArray = 0;
    gpuErrchk(cudaMalloc3DArray(&d_inputArray, &channelDesc, volumeSize));

    // --- Copy data to 3D array (host to device)
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(im->data, 
        im->nx * sizeof(float), im->nx, im->ny);
    copyParams.dstArray = d_inputArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    gpuErrchk(cudaMemcpy3D(&copyParams));

    // --- Set texture parameters
    tex.normalized = false; // access with un-normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear; // linear interpolation
    tex.addressMode[0] = cudaAddressModeBorder; // wrap texture coordinates
    tex.addressMode[1] = cudaAddressModeBorder;
    tex.addressMode[2] = cudaAddressModeBorder;

    // --- Bind array to 3D texture
    gpuErrchk(cudaBindTextureToArray(tex, d_inputArray, channelDesc));

    // --- Launch the interpolation kernel
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(DIVC(im->nx, blockSize.x), DIVC(im->ny, blockSize.y),
            DIVC(im->nz, blockSize.z));
    warp<<<gridSize, blockSize>>>(d_output, im->nx, im->ny, im->nz, xWarp,
        yWarp, zWarp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // --- Copy the interpolated data to the host, in-place
    gpuErrchk(cudaMemcpy(im->data,d_output,im_mem_size,cudaMemcpyDeviceToHost));

    CLEANUP
    return 0;

#undef CLEANUP
}
