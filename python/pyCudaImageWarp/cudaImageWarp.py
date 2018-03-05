"""
Warp an image using CUDA. Python wrapper for the C library.

(c) Blaine Rister 2018
"""

import numpy as np
import ctypes
import pyCudaImageWarp

"""
Arguments:
        im -- An image volume, i.e. a 3D numpy array. Indexed in Fortran order,
		e.g. im[x, y, z].
        A -- A [4x3] matrix defining the transformation. A[0, :] applies to the
		x-coordinates, A[1, :] the y-coordinates, A[2, :] the 
		z-coordinates. See im for more details.
        interp -- The interpolation type. Supported values are either 
                'linear' (default) or 'nearest'.
"""
def cudaImageWarp(im, A, interp='linear'):

	# Verify inputs
	ndim = 3;
	Ashape = (ndim, ndim + 1)
	if len(im.shape) != ndim:
		raise ValueError("im has shape %s, expected %d dimensions" % \
			(im.shape, ndim))
	if not np.equal(A.shape, Ashape).all():
		raise ValueError("Expected A shape %s, received %s" % \
			(Ashape, A.shape))

        # Mapping from interpolation strings to codes
        interpMap = {
                'nearest' : 0,
                'linear' : 1
        }

        # Convert the inputs to C float arrays
	dtype = im.dtype
        im = np.require(im, dtype='float32', requirements=['F', 'A', 'W', 'O'])
        A = np.require(A, dtype='float32', requirements=['C', 'A'])

        # Warp the image
        ret = pyCudaImageWarp.warpfun(
                im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(im.shape[0]), 
                ctypes.c_int(im.shape[1]), 
                ctypes.c_int(im.shape[2]), 
                ctypes.c_int(interpMap[interp]),
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        if ret != 0:
                raise ValueError(ret)

        # Convert the output back to the original type
        return im.astype(dtype)
