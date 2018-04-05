#ifndef __CUDA_IMAGE_WARP_H
#define __CUDA_IMAGE_WARP_H

#ifdef __cplusplus
extern "C" {
#endif

int cuda_image_warp(const float *const input, 
    const int nxi, const int nyi, const int nzi, 
    float *const output,
    const int nxo, const int nyo, const int nzo, 
    const int filter_mode, const float *const params, const float std,
    const float window_min, const float window_max);

#ifdef __cplusplus
}
#endif

#endif
