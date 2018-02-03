"""
Warp an image using CUDA. Python wrapper for the C library.

(c) Blaine Rister 2018
"""

import numpy as np
import ctypes
import pyCudaImageWarp


"""
Arguments:
        im -- An image volume, i.e. a 3D numpy array
        A -- A [4x3] matrix defining the transformation
"""
def cudaImageWarp(im, A):

        # Convert to float32
        im = im.astype(np.float32)
        A = A.astype(np.float32)

        # Warp the image
        ret = pyCudaImageWarp.warpfun(
                ctypes.c_void_p(im.ctypes.data), 
                ctypes.c_int(im.shape[0]), 
                ctypes.c_int(im.shape[1]), 
                ctypes.c_int(im.shape[2]), 
                ctypes.c_void_p(A.ctypes.data))

        if ret != 0:
                raise ValueError(ret)

        return im
