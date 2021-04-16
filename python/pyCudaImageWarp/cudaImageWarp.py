"""
Warp an image using CUDA. Python wrapper for the C library.

(c) Blaine Rister 2018
"""

import numpy as np
import ctypes
import pyCudaImageWarp

"""
Verify that the inputs are correct. Returns default parameter values.
"""
def __check_inputs(im, A, shape, device):

        # Make sure device is positive
        if device is None:
            # -1 signals no preference in device
            device = -1
        elif device < 0:
            raise ValueError(
                    "received device %d, must be non-negative!" % device)

        # Default to the same shape as im
        if shape is None:
            shape = im.shape

        # Check the dimensions
        ndim = 3;
        Ashape = (ndim, ndim + 1)
        if len(im.shape) != ndim:
                raise ValueError("im has shape %s, expected %d dimensions" % \
                        (im.shape, ndim))
        if len(shape) != ndim:
                raise ValueError("received output shape %s, expected %d "
                        "dimensions" % (shape, ndim))
        if not np.equal(A.shape, Ashape).all():
                raise ValueError("Expected A shape %s, received %s" % \
                        (Ashape, A.shape))

        return shape, device

""" 
Convert the inputs into the required formats for the C library.
"""
def __convert_inputs(im, A, interp):

        # Convert the interpolation string to and integer code
        interpMap = {
                'nearest' : 0,
                'linear' : 1
        }
        interpCode = interpMap[interp]

        # Convert the inputs to C float arrays
        im = np.require(im, dtype='float32', requirements=['F', 'A'])
        A = np.require(A, dtype='float32', requirements=['C', 'A'])

        return im, A, interpCode

"""
Create a C float array for the output
"""
def __create_output(shape):
        out = np.zeros(shape, dtype='float32')
        out = np.require(out, dtype='float32', 
                requirements=['F', 'A', 'W', 'O'])
        return out

"""
Shortcut to take care of inputs.
"""
def __handle_inputs(im, A, shape, interp, device):
        shape, device = __check_inputs(im, A, shape, device)
        im, A, interpCode = __convert_inputs(im, A, interp)
        return im, A, shape, interpCode, device

"""
Warp a since image. Returns the result in the same datatype as the input.

Arguments:
        im -- An image volume, i.e. a 3D numpy array. Indexed in Fortran order,
                e.g. im[x, y, z].
        A -- A [4x3] matrix defining the transformation. A[0, :] applies to the
                x-coordinates, A[1, :] the y-coordinates, A[2, :] the 
                z-coordinates. See im for more details.
        interp -- The interpolation type. Supported values are either 
                'linear' (default) or 'nearest'.
        shape -- The shape of the output. By default, this is the same as the 
                input. This can be used to crop or pad an image.
        std -- The standard derviation of white Gaussian noise added to the
                output.
        winMax -- The maximum intensity value to be used in the window.
        winMin -- The minimum intensity value to be used in the window.
        occZmin -- The minimum z-value to be occluded.
        occZmax -- The maximum z-value to be occluded.
        oob - The value assigned to out-of-bounds voxels.
        rounding - The rounding mode applied to the output. The options are:
                'nearest' - Python default float to input type conversion
                'positive' - 
        device -- The ID of the CUDA device to be used. (Optional)
"""
def warp(im, A, interp='linear', shape=None, std=0.0, 
        winMin=-float('inf'), winMax=float('inf'), occZmin=0, occZmax=-1, oob=0,
        rounding='default', device=None):

        # Handle inputs
        im, A, shape, interpCode, device = __handle_inputs(im, A, shape,
                interp, device)

        # Create the output
        out = __create_output(shape)

        # Warp
        ret = pyCudaImageWarp.warpfun(
                im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(im.shape[0]), 
                ctypes.c_int(im.shape[1]), 
                ctypes.c_int(im.shape[2]), 
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(shape[0]),
                ctypes.c_int(shape[1]),
                ctypes.c_int(shape[2]),
                ctypes.c_int(interpCode),
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(std),
                ctypes.c_float(winMin),
                ctypes.c_float(winMax),
                ctypes.c_int(occZmin),
                ctypes.c_int(occZmax),
                ctypes.c_float(oob),
                ctypes.c_int(device)
        )

        if ret != 0:
                raise ValueError(ret)

        # Returns in float32 format 
        return out


"""
Push an image onto the queue. See warp() for parameters.
"""
def push(im, A, interp='linear', shape=None, std=0.0, 
        winMin=-float('inf'), winMax=float('inf'), occZmin=0, occZmax=-1, 
        oob=0, rounding='default', device=None):

        # Handle inputs
        im, A, shape, interpCode, device = __handle_inputs(im, A, shape,
                interp, device)

        # Enqueue the image warping
        ret = pyCudaImageWarp.pushfun(
                im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(im.shape[0]), 
                ctypes.c_int(im.shape[1]), 
                ctypes.c_int(im.shape[2]), 
                ctypes.c_int(shape[0]),
                ctypes.c_int(shape[1]),
                ctypes.c_int(shape[2]),
                ctypes.c_int(interpCode),
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_float(std),
                ctypes.c_float(winMin),
                ctypes.c_float(winMax),
                ctypes.c_int(occZmin),
                ctypes.c_int(occZmax),
                ctypes.c_float(oob),
                ctypes.c_int(device)
        )

        if ret != 0:
                raise ValueError(ret)

        # Push the inputs onto the queue
        pyCudaImageWarp.q.put({
                'shape': shape,
                'rounding': rounding
        })

""" 
Finish processing the top image on the queue, returning the result.
""" 
def pop():

        # Retrieve the inputs
        inputs = pyCudaImageWarp.q.get_nowait()
        
        # Create the output array
        out = __create_output(inputs['shape'])

        # Get the result
        ret = pyCudaImageWarp.popfun(
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
                
        
        if ret != 0:
                raise ValueError(ret)

        # Returns in float32 format 
        return out
