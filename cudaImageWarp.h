#ifndef __CUDA_IMAGE_WARP_H
#define __CUDA_IMAGE_WARP_H

#ifdef __cplusplus
extern "C" {
#endif

int cuda_image_warp(Image *const im, const float *const params);

#ifdef __cplusplus
}
#endif

#endif
