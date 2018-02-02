#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <sift3d/imtypes.h>
#include <sift3d/imutil.h>

#define DIVC(x, y)  (((x) + (y) + 1) / (y)) // Divide integers and ceil
#define AFFINE_WARP(x, y, z, f4) /* Warp using a float4 */ \
        (x * f4.x + y * f4.y + z * f4.z + f4.w)

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}

typedef unsigned int  uint;

texture<float, 3, cudaReadModeElementType> tex;  // 3D texture

/************************************************/
/* TEXTURE-BASED TRILINEAR INTERPOLATION KERNEL */
/************************************************/
__global__ void
d_render(float *d_output, const uint nx, const uint ny, const uint nz, 
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

int main(int argc, char *argv[]) {

    Image im;

    // The transform parameters
    const float4 xWarp = {1.0, 0.0, 0.0, 0.0};
    const float4 yWarp = {0.0, 0.5, 0.0, 0.0};
    const float4 zWarp = {0.0, 0.0, 2.0, 0.0};

    const int num_args = 1;

    // Get the inputs
    if (argc != num_args + 1) {
        fprintf(stderr, "This program takes %d argument(s)\n", num_args);
        return -1;
    }

    const char* filename = argv[1];

    // Initialize the image
    init_im(&im);

    // --- Loading data from file
    if (im_read(filename, &im)) {
        fprintf(stderr, "Error reading file %s\n", filename);
        return -1;
    }

    // --- Allocate device memory for output
    float *d_output = NULL;
    const size_t im_mem_size = im.size * sizeof(float);
    gpuErrchk(cudaMalloc((void**)&d_output, im_mem_size));

    // --- Create 3D array
    const cudaExtent volumeSize = make_cudaExtent(im.nx, im.ny, im.nz);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *d_inputArray = 0;
    gpuErrchk(cudaMalloc3DArray(&d_inputArray, &channelDesc, volumeSize));

    // --- Copy data to 3D array (host to device)
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(im.data, 
        im.nx * sizeof(float), im.nx, im.ny);
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
    const dim3 gridSize(DIVC(im.nx, blockSize.x), DIVC(im.ny, blockSize.y),
            DIVC(im.nz, blockSize.z));
    d_render<<<gridSize, blockSize>>>(d_output, im.nx, im.ny, im.nz, xWarp,
        yWarp, zWarp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // --- Copy the interpolated data to the host, in-place
    gpuErrchk(cudaMemcpy(im.data,d_output,im_mem_size,cudaMemcpyDeviceToHost));

    // Write to a file
    im_write("out_image.nii.gz", &im); 

    return 0;
}
