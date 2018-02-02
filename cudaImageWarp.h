#ifndef __CUDA_IMAGE_WARP_H
#define __CUDA_IMAGE_WARP_H

#ifdef __cplusplus
extern "C" {
#endif

int cuda_image_warp(float *const im_data, const int nx, const int ny, 
        const int nz, const float *const params);

#ifdef __cplusplus
}
#endif

#endif
