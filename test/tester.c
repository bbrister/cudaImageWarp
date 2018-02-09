/* Test program for CUDA image warper */

#include <stdio.h>

#include <cuda_runtime.h>

#include <sift3d/imtypes.h>
#include <sift3d/imutil.h>

#include "cudaImageWarp.h"

int main(int argc, char *argv[]) {

    Image im;

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

    // Initialize the image
    init_im(&im);

    // Read the image from a file
    if (im_read(filename, &im)) {
        fprintf(stderr, "Error reading file %s\n", filename);
        return -1;
    }

    if (im.nc != 1) {
        fputs("Multi-channel images are not supported\n", stderr);
        return -1;
    }

    // Warp the image
    if (cuda_image_warp(im.data, im.nx, im.ny, im.nz, 1, params)) {
        fputs("Error warping image", stderr);
        return -1;
    }

    // Write to a file
    return im_write("out_image.nii.gz", &im); 
}
