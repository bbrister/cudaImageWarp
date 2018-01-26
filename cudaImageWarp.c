#include <stdio.h>
#include <stdlib.h>
#include <fstream> 

#include <cuda_runtime.h>
#include <cuda.h>

typedef unsigned char uchar;

#define BLOCKSIZE 16

float w = 0.5;  // texture coordinate in z

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}

typedef unsigned int  uint;
typedef unsigned char uchar;

texture<uchar, 3, cudaReadModeNormalizedFloat> tex;  // 3D texture

cudaArray *d_volumeArray = 0;

uint *d_output = NULL;
uint *h_output = NULL;

/************************************************/
/* TEXTURE-BASED TRILINEAR INTERPOLATION KERNEL */
/************************************************/
__global__ void
d_render(uint *d_output, uint imageW, uint imageH, float w)
{
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    float u = x / (float) imageW;
    float v = y / (float) imageH;

    // read from 3D texture
    float voxel = tex3D(tex, u, v, w);

    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = __umul24(y, imageW) + x;
        d_output[i] = voxel*255;
   }
}

void main() {

    int N = 32;
    int imageH = 512;
    int imageW = 512;

    const char* filename = "Bucky.raw";

    // --- Loading data from file
    FILE *fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Error opening file '%s'\n", filename); getchar(); return; }

    uchar *data = (uchar*)malloc(N*N*N*sizeof(uchar));
    size_t read = fread(data, 1, N*N*N, fp);
    fclose(fp);

    printf("Read '%s', %lu bytes\n", filename, read);

    gpuErrchk(cudaMalloc((void**)&d_output, imageH*imageW*sizeof(uint)));

    // --- Create 3D array
    const cudaExtent volumeSize = make_cudaExtent(N, N, N);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    gpuErrchk(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // --- Copy data to 3D array (host to device)
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)data, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    gpuErrchk(cudaMemcpy3D(&copyParams));

    // --- Set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.addressMode[2] = cudaAddressModeWrap;

    // --- Bind array to 3D texture
    gpuErrchk(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

    // --- Launch the interpolation kernel
    const dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);
    const dim3 gridSize(imageW / blockSize.x, imageH / blockSize.y);
    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, w);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // --- Copy the interpolated data to host
    h_output = (uint*)malloc(imageW*imageH*sizeof(uint));
    gpuErrchk(cudaMemcpy(h_output,d_output,imageW*imageH*sizeof(uint),cudaMemcpyDeviceToHost));

    std::ofstream outfile;
    outfile.open("out_texture.dat", std::ios::out | std::ios::binary);
    outfile.write((char*)h_output, imageW*imageH*sizeof(uint));
    outfile.close();

    getchar();

}
