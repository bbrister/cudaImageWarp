#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cudaImageWarp.h" // Need this to get C linkage on exported functions

#define DIVC(x, y)  (((x) + (y) + 1) / (y)) // Divide positive integers and ceil

/* Global data */
typedef unsigned int  uint;

/* CUDA device function which does the affine warping */
__device__ float affine_warp(const uint x, const uint y, const uint z, 
	const float4 warp) {
        return x * warp.x + y * warp.y + z * warp.z + warp.w;
}

/* Sample from a texture after affine warping */
__device__ float sample_affine(cudaTextureObject_t tex,
    float *const output, 
    const uint x, const uint y, const uint z,
    const float4 xWarp, const float4 yWarp, const float4 zWarp) {

    const float xs = affine_warp(x, y, z, xWarp);
    const float ys = affine_warp(x, y, z, yWarp);
    const float zs = affine_warp(x, y, z, zWarp);

    return tex3D<float>(tex, xs, ys, zs);
}

/* Deice function to perform image post-processing */
__device__ float postprocess(const float in, curandState_t *state, 
	const float std, const float window_min, const float window_max) {

    // Generate white Gaussian noise, if std > 0
    const float noise = std > 0 ? curand_normal(state) * std : 0.0f;

    // Add the noise
    float out = in + noise;

    // Clamp using the window bounds
    out = max(out, window_min);
    out = min(out, window_max);

    // Normalize to [0,1] using the window
    const float window_width = window_max - window_min;
    return isfinite(window_width) ? (out - window_min) / window_width : out;
}

/* Image warping kernel */
__global__ void
warp(cudaTextureObject_t tex, float *const output, curandState_t *const states,
    const float std, const uint nx, const uint ny, const uint nz, 
    const float window_min, const float window_max,
    const float4 xWarp, const float4 yWarp, const float4 zWarp)
{

#define CUDA_SET_DIMS \
    const uint x = blockIdx.x * blockDim.x + threadIdx.x; \
    const uint y = blockIdx.y * blockDim.y + threadIdx.y; \
    const uint z = blockIdx.z * blockDim.z + threadIdx.z; \
    \
    /* Check boundaries */ \
    if (x >= nx || y >= ny || z >= nz) \
	return; \
    \
    const uint y_stride = nx; \
    const uint z_stride = nx * ny; \
    const uint idx = z * z_stride + y * y_stride + x; 

    CUDA_SET_DIMS

    // Read from the 3D texture and postprocess
    const float in = sample_affine(tex, output, x, y, z, xWarp, yWarp, zWarp);
    output[idx] =  postprocess(in, states + idx, std, window_min, window_max);
}

/* Initialize an RNG for each thread. This uses a different seed for each
 * generator. This is much faster than using separate sequences with the same
 * seed, but we are not guaranteed independence between generators. */
__global__ void initRand(const int seed, curandState_t *const states,
    const uint nx, const uint ny, const uint nz) {
    CUDA_SET_DIMS // See warp()
    curand_init(seed + idx, 0, 0, states + idx);
}

/* Warp an image in-place.  
* Parameters:
*  input - an array of nxi * nyi * nzi floats, strided in (x,y,z) order
*  nxi, nyi, nzi - the input image dimensions
*  output - an array of nxo * nyo * nzo floats, strided in (x,y,z) order
*  nxo, nyo, nzo - the output image dimensions
*  filter_mode - use 0 for nearest neighbor, 1 for linear
*  params - an array of 12 floats, in row-major order
*  std - standard deviation for additive white Gaussian noise. Disables noise if
*	std <= 0.
*  window_min - the minimum value for the window. Use -INFINITY to do nothing.
*  window_max - the maximum value for the window. Use INFINITY to do nothing.
*
* Returns 0 on success, nonzero otherwise. */
int cuda_image_warp(const float *const input, 
    const int nxi, const int nyi, const int nzi, 
    float *const output,
    const int nxo, const int nyo, const int nzo, 
    const int filter_mode, const float *const params, const float std,
    const float window_min, const float window_max) {

    // Convert the input to CUDA datatypes
    const float4 xWarp = {params[0], params[1], params[2], params[3]};
    const float4 yWarp = {params[4], params[5], params[6], params[7]};
    const float4 zWarp = {params[8], params[9], params[10], params[11]}; 

    // Intermediates
    float *d_output = NULL;
    curandState_t *d_states = NULL;
    cudaArray *d_input = NULL;

#define gpuAssert(code, file, line) { \
if (code != cudaSuccess) { \
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), \
	    file, line); \
    if (d_output != NULL) \
	cudaFree(d_output); \
    if (d_states != NULL) \
	cudaFree(d_states); \
    if (d_input != NULL) \
	cudaFreeArray(d_input); \
    return -1; \
}  \
} \

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define CLEANUP { \
if (d_output != NULL) { \
    cudaFree(d_output); \
    gpuErrchk(cudaPeekAtLastError()); \
} \
if (d_states != NULL) { \
    cudaFree(d_states); \
    gpuErrchk(cudaPeekAtLastError()); \
} \
if (d_input != NULL) { \
    cudaFreeArray(d_input); \
    gpuErrchk(cudaPeekAtLastError()); \
} \
} 

    // --- Allocate device memory for output
    const size_t num_voxels = nxo * nyo * nzo;
    const size_t out_size = num_voxels * sizeof(float);
    gpuErrchk(cudaMalloc((void**) &d_output, out_size));
    gpuErrchk(cudaMalloc((void**) &d_states, 
	num_voxels * sizeof(curandState_t)));

    // --- Create 3D array
    const cudaExtent inVolumeSize = make_cudaExtent(nxi, nyi, nzi);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    gpuErrchk(cudaMalloc3DArray(&d_input, &channelDesc, inVolumeSize));

    // --- Copy the input data to a 3D array (host to device)
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void *) input,
	nxi * sizeof(float), nxi, nyi);
    copyParams.dstArray = d_input;
    copyParams.extent   = inVolumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    gpuErrchk(cudaMemcpy3D(&copyParams));

    // --- Create the texture object
    cudaTextureObject_t tex;
    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = d_input;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.addressMode[0] = cudaAddressModeBorder;
    texDescr.addressMode[1] = cudaAddressModeBorder;
    texDescr.addressMode[2] = cudaAddressModeBorder;
    texDescr.readMode = cudaReadModeElementType;
    switch (filter_mode) {
	case 0:
	    // Nearest neighbor interpolation
	    texDescr.filterMode = cudaFilterModePoint;
	    break;
	case 1:
	    // Linear interpolation
	    texDescr.filterMode = cudaFilterModeLinear;
	    break;
	default:
	    fprintf(stderr, "Unrecognized filter_mode: %d \n", filter_mode);
            CLEANUP
	    return -1;
    }
    gpuErrchk(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    // Configure the block and grid sizes
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(DIVC(nxo, blockSize.x), DIVC(nyo, blockSize.y),
	DIVC(nzo, blockSize.z));

    // Initialize the random number generators
    //TODO we could keep track of the last image size globally, only
    // calling this kernel if that number changes
    if (std > 0.0f) { 
	// Get the random seed from the time
	const time_t seed = clock();

	// Initialize one RNG per thread
	initRand<<<gridSize, blockSize>>>(seed, d_states, nxo, nyo, nzo);
    }

    // Perform image warping and augmentation
    warp<<<gridSize, blockSize>>>(tex, d_output, d_states, std, nxo, nyo, nzo, 
	window_min, window_max, xWarp, yWarp, zWarp);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // --- Copy the output data to the host
    gpuErrchk(cudaMemcpy(output,d_output,out_size,cudaMemcpyDeviceToHost));

    // Destroy the texture object
    cudaDestroyTextureObject(tex);  

    CLEANUP
    return 0;

#undef CLEANUP
}

