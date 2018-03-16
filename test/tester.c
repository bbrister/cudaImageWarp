/* Test program for CUDA image warper */

#include <stdio.h>

#include <cuda_runtime.h>

#include <sift3d/imtypes.h>
#include <sift3d/imutil.h>

#include "cudaImageWarp.h"

int main(int argc, char *argv[]) {

    Image in, out;

    // The transform parameters
    const float params[] = {1.0, 0.0, 0.0, 0.0,
                            0.0, 0.5, 0.0, 0.0,
                            0.0, 0.0, 2.0, 0.0};

    const int num_args = 1;

    // Get the inputs
    if (argc != num_args + 1) {
        fprintf(stderr, "This program takes %d argument(s)\n", num_args);
        return -1;
    }

    const char* filename = argv[1];

    // Initialize the input image
    init_im(&in);

    // Read the image from a file
    if (im_read(filename, &in)) {
        fprintf(stderr, "Error reading file %s\n", filename);
        return -1;
    }

    if (in.nc != 1) {
        fputs("Multi-channel images are not supported\n", stderr);
        return -1;
    }

    // Initialize the output image
    init_im_with_dims(&out, in.nx, in.ny, in.nz, in.nc);

    // Warp the image
    if (cuda_image_warp(in.data, in.nx, in.ny, in.nz, 
        out.data, out.nx, out.ny, out.nz, 1, params)) {
        fputs("Error warping image", stderr);
        return -1;
    }

    // Write to a file
    return im_write("out_image.nii.gz", &out); 
}
