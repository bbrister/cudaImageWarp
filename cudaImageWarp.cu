#include <queue> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* Core CUDA headers */
#include <cuda_runtime.h>
#include <cuda.h>

/* Optional CURAND headers */
#ifdef WITH_CURAND
#include <curand.h>
#include <curand_kernel.h>
#else
        /* Dummy definitions for non-curand version */
        typedef int curandState_t;
        __device__ void curand_init(int a, int b, int c, curandState_t *state) {}
        __device__ float curand_normal(curandState_t *state) { return 0.0; }
#endif

#include "cudaImageWarp.h" // Need this to get C linkage on exported functions

/* Keep track of the most recently used device */
int current_device = 0;

/* Useful macros */
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define DIVC(x, y)  (((x) + (y) + 1) / (y)) // Divide positive integers and ceil

#define gpuAssert(code, file, line) { \
if (code != cudaSuccess) { \
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), \
	    file, line); \
    return -1; \
}  \
} \

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* CUDA device function which does the affine warping */
__device__ float affine_warp(const int x, const int y, const int z, 
    const float4 warp) {
    return x * warp.x + y * warp.y + z * warp.z + warp.w;
}

/* Device function to perform image post-processing */
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
warp(cudaTextureObject_t tex, float *const output, curandState_t *const rands,
    const float std, const int nxi, const int nyi, const int nzi, 
    const int nxo, const int nyo, const int nzo, 
    const float window_min, const float window_max,
    const float4 xWarp, const float4 yWarp, const float4 zWarp, 
    const int occZmin, const int occZmax, const float oob)
{

#define CUDA_SET_DIMS \
    const int x = blockIdx.x * blockDim.x + threadIdx.x; \
    const int y = blockIdx.y * blockDim.y + threadIdx.y; \
    const int z = blockIdx.z * blockDim.z + threadIdx.z; \
    \
    /* Check boundaries */ \
    if (x >= nxo || y >= nyo || z >= nzo) \
	return; \
    \
    const int y_stride = nxo; \
    const int z_stride = nxo * nyo; \
    const int idx = z * z_stride + y * y_stride + x; 

    CUDA_SET_DIMS

    // Perform occlusion, exit early for occluded voxels
    if (z >= occZmin && z <= occZmax) {
        output[idx] = oob;
        return;
    }

    // Compute the affine sampling coordinates
    const float xs = affine_warp(x, y, z, xWarp);
    const float ys = affine_warp(x, y, z, yWarp);
    const float zs = affine_warp(x, y, z, zWarp);

    // Check for out-of-bounds texture access
    const float nxif = (float) nxi + 1.0f;
    const float nyif = (float) nyi + 1.0f;
    const float nzif = (float) nzi + 1.0f;
    if (xs < 0 || xs > nxif || ys < 0 || ys > nyif || zs < 0 || zs > nzif) {
        output[idx] = oob;
        return;
    }

    // Sample from the 3D texture
    const float in = tex3D<float>(tex, xs + 0.5, ys + 0.5, zs + 0.5);

    // Postprocess
    output[idx] = postprocess(in, rands + idx, std, window_min, window_max);
}

/* Initialize an RNG for each thread. This uses a different seed for each
 * generator. This is much faster than using separate sequences with the same
 * seed, but we are not guaranteed independence between generators. */
__global__ void initRand(const int seed, curandState_t *const rands,
    const int nxo, const int nyo, const int nzo) {
    CUDA_SET_DIMS // See warp()
    curand_init(seed + idx, 0, 0, rands + idx);
}

/* Function prototypes */
static void cudaFreeAndNull(void *&ptr);
static void cudaFreeArrayAndNull(cudaArray *&ptr);
static size_t get_num_voxels(const int nx, const int ny, const int nz);
static size_t get_size(const int nx, const int ny, const int nz);
static void affine_image(const float4 warp, const int nxo, const int nyo, 
	const int nzo, const int clamp, int *const min, int *const max);

/* Class for handling the state */
class State {

private:
    cudaTextureObject_t tex; // Input texture
    void *output; // Device output buffer (float)
    curandState_t *rands; // Random number generators
    cudaArray *input; // Device input array
    size_t output_size; // The size of output, in bites
    int have_tex; // True if tex was created

    /* Free any resources which were allocated. */
    int cleanup(void) {
        
        // Free the input memory
        cudaFreeArrayAndNull(input);
        gpuErrchk(cudaPeekAtLastError());

        // Free the output memory
        cudaFreeAndNull(output);
        gpuErrchk(cudaPeekAtLastError());

        // Free random number generators
        void *rands_v = static_cast<void *>(rands);
        cudaFreeAndNull(rands_v);
        gpuErrchk(cudaPeekAtLastError());
        rands = NULL;

        // Destroy the texture object
        if (have_tex) {
            cudaDestroyTextureObject(tex);
            gpuErrchk(cudaPeekAtLastError());
            have_tex = 0;
        }

        return 0;
    }

    /* The meat of warp_start, wrapped to handle errors. */
    int _warp_start(const float *const input,
        const int nxi, const int nyi, const int nzi, 
        const int nxo, const int nyo, const int nzo, 
        const int filter_mode, const float *const params, const float std,
        const float window_min, const float window_max, const int occZmin, 
	const int occZmax, const float oob, const int device) {

	// Settings
	const int with_rands = std > 0.0f;

        // Check if this was compiled with curand
#ifndef WITH_CURAND
        if (with_rands) {
                fprintf(stderr, "Received std %f, but library was compiled "
                        "without cuRand!", std);
                return -1;
        }
#endif

        // Convert the input to CUDA datatypes
        const float4 xWarp = {params[0], params[1], params[2], params[3]};
        const float4 yWarp = {params[4], params[5], params[6], params[7]};
        const float4 zWarp = {params[8], params[9], params[10], params[11]}; 

	// Get the number of devices, and choose the next one for use
	int num_devices;
	gpuErrchk(cudaGetDeviceCount(&num_devices));
        current_device = (device < 0 ? current_device + 1 : device) % 
            num_devices;
	cudaSetDevice(current_device);

	// --- Allocate device memory for output
	const size_t num_voxels = get_num_voxels(nxo, nyo, nzo);
	output_size = get_size(nxo, nyo, nzo);
	gpuErrchk(cudaMalloc((void**) &output, output_size));
	if (with_rands) {
	    gpuErrchk(cudaMalloc((void**) &rands,
	    num_voxels * sizeof(curandState_t)));
	}

	/* Get a sub-volume of the input containing the image of the output 
	 * under the affine transformation. This is all we need to send to the
	 * GPU. */
	int x_min, y_min, z_min, x_max, y_max, z_max;
	affine_image(xWarp, nxo, nyo, nzo, nxi - 1, &x_min, &x_max);
	affine_image(yWarp, nxo, nyo, nzo, nyi - 1, &y_min, &y_max);
	affine_image(zWarp, nxo, nyo, nzo, nzi - 1, &z_min, &z_max);

	// Translate the existing affine warps for the sub-volume
	const float4 xWarpSub = {xWarp.x, xWarp.y, xWarp.z, xWarp.w - x_min};
	const float4 yWarpSub = {yWarp.x, yWarp.y, yWarp.z, yWarp.w - y_min};
	const float4 zWarpSub = {zWarp.x, zWarp.y, zWarp.z, zWarp.w - z_min};

        // Create 3D array for the sub-volume
	const int nxis = x_max - x_min + 1;  
	const int nyis = y_max - y_min + 1;  
	const int nzis = z_max - z_min + 1;
        const cudaExtent texSize = make_cudaExtent(nxis, nyis, nzis);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	gpuErrchk(cudaMalloc3DArray(&this->input, &channelDesc, 
					texSize));

        /* Copy the input data to a 3D array (host to device). Note this uses
	 * the address strides of the original input volume, not the 
	 * sub-volume. */
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr((void *) input,
            nxi * sizeof(float), nxi, nyi);
	copyParams.srcPos   = make_cudaPos(x_min,
					   y_min, 
					   z_min);
        copyParams.dstArray = this->input;
        copyParams.extent   = texSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        gpuErrchk(cudaMemcpy3D(&copyParams));

        // --- Create the texture object
        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array  = this->input;
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
                // Tri-linear interpolation
                texDescr.filterMode = cudaFilterModeLinear;
                break;
            default:
                fprintf(stderr, "Unrecognized filter_mode: %d \n", filter_mode);
                return -1;
        }
        gpuErrchk(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
        have_tex = 1;

        // Configure the block and grid sizes
        const dim3 blockSize(16, 16, 1);
        const dim3 gridSize(DIVC(nxo, blockSize.x), DIVC(nyo, blockSize.y),
            DIVC(nzo, blockSize.z));

        // Initialize the random number generators
        //TODO we could keep track of the last image size globally, only
        // calling this kernel if that number changes
        if (with_rands) { 
            // Get the random seed from the time
            const time_t seed = clock();

            // Initialize one RNG per thread
            initRand<<<gridSize, blockSize>>>(seed, rands, nxo, nyo, nzo);
        }

        // Perform image warping and augmentation
        warp<<<gridSize, blockSize>>>(tex, (float *) output, 
            rands, std, nxis, nyis, nzis, nxo, nyo, nzo, window_min, window_max, 
	    xWarpSub, yWarpSub, zWarpSub, occZmin, occZmax, oob);
        gpuErrchk(cudaPeekAtLastError());

        return 0;
    }

public:
    State(void) : output(NULL), rands(NULL), input(NULL), have_tex(0) {};

    /* Finish computation and copy the result to host. Cleans up the state. */
    int warp_finish(float *const output) {

        // --- Copy the output data to the host
        gpuErrchk(cudaMemcpy(output, this->output, output_size, 
            cudaMemcpyDeviceToHost));

        return cleanup();
    }

    /* Start processing an image. */
    int warp_start(const float *const input,
        const int nxi, const int nyi, const int nzi, 
        const int nxo, const int nyo, const int nzo, 
        const int filter_mode, const float *const params, const float std,
        const float window_min, const float window_max, const int occZmin, 
	const int occZmax, const float oob, const int device) {

        int ret;

        if ((ret = _warp_start(input, nxi, nyi, nzi, nxo, nyo, nzo, filter_mode,
                params, std, window_min, window_max, occZmin, occZmax, oob, 
		device)) != 0) {
            cleanup();
        }

        return ret;
    }
};

/* Queue to store the state */
std::queue<State> q;

/* Call cudaFree, then set to NULL */
static void cudaFreeAndNull(void *&ptr) {
    if (ptr == NULL) return;
    cudaFree(ptr);
    ptr = NULL;
}

/* Like cudaFreeAndNull, for but CudaArrays */
static void cudaFreeArrayAndNull(cudaArray *&ptr) {
    if (ptr == NULL) return;
    cudaFreeArray(ptr);
    ptr = NULL; 
}

/* Compute the number of voxels in an array */
static size_t get_num_voxels(const int nx, const int ny, const int nz) {
    return ((size_t) nx) * ((size_t) ny) * ((size_t) nz);
}

/* Compute the size of an array, in the internal represenataion */
static size_t get_size(const int nx, const int ny, const int nz) {
    return get_num_voxels(nx, ny, nz) * sizeof(float);
}

/* This is the host version of affine_warp. */
static float affine_warp_host(const float4 warp, int x, int y, int z) {
    return warp.x * x + warp.y * y + warp.z * z + warp.w;
} 

/* Compute the extreme points of the image of the input volume under the
 * affine functional 'warp'. Return these in min and max. Return values are
 * clamped to the range (0, clamp). */
static void affine_image(const float4 warp, const int nxo, const int nyo, 
	const int nzo, const int clamp, int *const min, int *const max) {

    int x, y, z;

    // Initialize the result to extreme values
    int running_min = clamp;
    int running_max = 0; 

    // Warp all 8 corners of the cube	
    for (z = 0; z < 2; z++) {
    for (y = 0; y < 2; y++) {
    for (x = 0; x < 2; x++) {
	// Get the corner coordinate	
	const int xc = x > 0 ? nxo - 1 : 0;
	const int yc = y > 0 ? nyo - 1 : 0;
	const int zc = z > 0 ? nzo - 1 : 0;

	// Compute the warp and round
	const float val = affine_warp_host(warp, xc, yc, zc);
	const int lo = (int) floorf(val);
	const int hi = lo + 1;

	// Adjust the running min/max
	if (lo < running_min) running_min = lo;
	if (hi > running_max) running_max = hi;
    }}}

    // Save the results
    *min = MAX(MIN(running_min, clamp - 1), 0);
    *max = MAX(MIN(running_max, clamp), 0);
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
*  occZmin - the minimum z-coordinate to be occluded
*  occZmax - the maximimum z-coordinate to be occluded
*  oob - the value for out-of-bounds voxels
*  device - the index of the CUDA device to use. If negative, defaults to the
	least recently used device.
*
*  Note: This does not interfere with the queue.
*
* Returns 0 on success, nonzero otherwise. */
int cuda_image_warp(const float *const input, 
    const int nxi, const int nyi, const int nzi, 
    float *const output,
    const int nxo, const int nyo, const int nzo, 
    const int filter_mode, const float *const params, const float std,
    const float window_min, const float window_max,
    const int occZmin, const int occZmax, const float oob, const int device) {

    State state;

    return state.warp_start(input, nxi, nyi, nzi, nxo, 
            nyo, nzo, filter_mode, params, std, window_min, window_max, occZmin,
            occZmax, oob, device) || state.warp_finish(output);
                
}

/* Call cuda_image_warp_finish and pop the state from the queue. */
int cuda_image_warp_pop(float *const output) {

    int ret;

    // Finish computation
    State &state = q.front();
    ret = state.warp_finish(output);

    // Pop from the queue
    q.pop();
    return ret;
}

/* Enqueue an image warping. See cuda_image_warp for parameters. Get the 
 * reuslt via cuda_image_warp_pop. */
int cuda_image_warp_push(const float *const input,
    const int nxi, const int nyi, const int nzi, 
    const int nxo, const int nyo, const int nzo, 
    const int filter_mode, const float *const params, const float std,
    const float window_min, const float window_max,
    const int occZmin, const int occZmax, const float oob, const int device) {

    int ret;

    // Create a new state
    State state;

    // Start the computation
    if ((ret = state.warp_start(input, nxi, nyi, nzi, nxo, nyo, 
        nzo, filter_mode, params, std, window_min, window_max, occZmin, 
        occZmax, oob, device)))
        return ret;
        
    // Add the state to the queue
    q.push(state);
    return 0;
}

