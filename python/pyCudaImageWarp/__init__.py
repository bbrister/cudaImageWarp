
"""
Python module initializer for CUDA image warper.

(c) Blaine Rister 2018
"""

import os
import ctypes

# Get the script location
scriptDir = os.path.dirname(os.path.abspath(__file__))
libDir = os.path.join(scriptDir, '..', '..', 'build', 'lib')
libName = os.path.join(libDir, 'libcudaImageWarp.so')

# Load the library
dll = ctypes.cdll.LoadLibrary(libName)

# Extract the warping function and set up its signature
warpfun = dll.cuda_image_warp
warpfun.argtypes = [
        ctypes.c_void_p, 
        ctypes.c_int, 
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
        ]
warpfun.restype = ctypes.c_int
