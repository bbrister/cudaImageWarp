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
        interp -- The interpolation type. Supported values are either 
                'linear' (default) or 'nearest'.
"""
def cudaImageWarp(im, A, interp='linear'):

        # Mapping from interpolation strings to codes
        interpMap = {
                'nearest' : 0,
                'linear' : 1
        }

        # Copy the image and extract its C interface
        imFloat = np.copy(im).astype(np.float32)
        AFloat = A.astype(np.float32)
        imC = imFloat.ctypes
        AC = AFloat.ctypes

        # Warp the image
        ret = pyCudaImageWarp.warpfun(
                imC.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(im.shape[0]), 
                ctypes.c_int(im.shape[1]), 
                ctypes.c_int(im.shape[2]), 
                ctypes.c_int(interpMap[interp]),
                AC.data_as(ctypes.POINTER(ctypes.c_float))
        )

        if ret != 0:
                raise ValueError(ret)

        # Save the image as the original type
        return imFloat.astype(im.dtype)
