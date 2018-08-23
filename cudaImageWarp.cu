#include <queue>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cudaImageWarp.h" // Need this to get C linkage on exported functions

/* Useful macros */
#define DIVC(x, y)  (((x) + (y) + 1) / (y)) // Divide positive integers and ceil

#define gpuAssert(code, file, line) { \
if (code != cudaSuccess) { \
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), \
	    file, line); \
    return -1; \
}  \
} \

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* Types */
typedef unsigned int uint;

/* CUDA device function which does the affine warping */
__device__ float affine_warp(const uint x, const uint y, const uint z, 
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
    const float std, const uint nx, const uint ny, const uint nz, 
    const float window_min, const float window_max,
    const float4 xWarp, const float4 yWarp, const float4 zWarp, 
    const int occZmin, const int occZmax)
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

    // Perform occlusion, exit early for occluded voxels
    if (z >= occZmin && z <= occZmax) {
        output[idx] = 0;
        return;
    }

    // Compute the affine sampling coordinates
    const float xs = affine_warp(x, y, z, xWarp);
    const float ys = affine_warp(x, y, z, yWarp);
    const float zs = affine_warp(x, y, z, zWarp);

    // Sample from the 3D texture
    const float in = tex3D<float>(tex, xs, ys, zs);

    // Postprocess
    output[idx] = postprocess(in, rands + idx, std, window_min, window_max);
}

/* Initialize an RNG for each thread. This uses a different seed for each
 * generator. This is much faster than using separate sequences with the same
 * seed, but we are not guaranteed independence between generators. */
__global__ void initRand(const int seed, curandState_t *const rands,
    const uint nx, const uint ny, const uint nz) {
    CUDA_SET_DIMS // See warp()
    curand_init(seed + idx, 0, 0, rands + idx);
}

/* Function prototypes */
static void cudaFreeAndNull(void *&ptr);
static void cudaFreeArrayAndNull(cudaArray *&ptr);
static size_t get_num_voxels(const int nx, const int ny, const int nz);
static size_t get_size(const int nx, const int ny, const int nz);

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
        const float window_min, const float window_max,
        const int occZmin, const int occZmax) {

        // Convert the input to CUDA datatypes
        const float4 xWarp = {params[0], params[1], params[2], params[3]};
        const float4 yWarp = {params[4], params[5], params[6], params[7]};
        const float4 zWarp = {params[8], params[9], params[10], params[11]}; 

        // --- Allocate device memory for output
        const size_t num_voxels = get_num_voxels(nxo, nyo, nzo);
        output_size = get_size(nxo, nyo, nzo);
        gpuErrchk(cudaMalloc((void**) &output, output_size));
        gpuErrchk(cudaMalloc((void**) &rands, 
            num_voxels * sizeof(curandState_t)));

        // --- Create 3D array
        const cudaExtent inVolumeSize = make_cudaExtent(nxi, nyi, nzi);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            gpuErrchk(cudaMalloc3DArray(&this->input, &channelDesc, inVolumeSize));

        // --- Copy the input data to a 3D array (host to device)
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr((void *) input,
            nxi * sizeof(float), nxi, nyi);
        copyParams.dstArray = this->input;
        copyParams.extent   = inVolumeSize;
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
                // Linear interpolation
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
        if (std > 0.0f) { 
            // Get the random seed from the time
            const time_t seed = clock();

            // Initialize one RNG per thread
            initRand<<<gridSize, blockSize>>>(seed, rands, nxo, nyo, nzo);
        }

        // Perform image warping and augmentation
        warp<<<gridSize, blockSize>>>(tex, (float *) output, 
            rands, std, nxo, nyo, nzo, window_min, window_max, xWarp, 
            yWarp, zWarp, occZmin, occZmax);
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
        const float window_min, const float window_max,
        const int occZmin, const int occZmax) {

        int ret;

        if ((ret = _warp_start(input, nxi, nyi, nzi, nxo, nyo, nzo, filter_mode,
                params, std, window_min, window_max, occZmin, occZmax)) != 0) {
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
    const int occZmin, const int occZmax) {

    State state;

    return state.warp_start(input, nxi, nyi, nzi, nxo, 
            nyo, nzo, filter_mode, params, std, window_min, window_max, occZmin,
            occZmax) || state.warp_finish(output);
                
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
    const int occZmin, const int occZmax) {

    int ret;

    // Create a new state
    State state;

    // Start the computation
    if ((ret = state.warp_start(input, nxi, nyi, nzi, nxo, nyo, 
        nzo, filter_mode, params, std, window_min, window_max, occZmin, 
        occZmax)))
        return ret;
        
    // Add the state to the queue
    q.push(state);
    return 0;
}

