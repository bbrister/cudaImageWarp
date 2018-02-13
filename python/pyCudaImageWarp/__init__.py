
"""
Python module initializer for CUDA image warper.

(c) Blaine Rister 2018
"""

import os
import ctypes

# Load the library
libName = 'libcudaImageWarp.so'
scriptDir = os.path.abspath(os.path.dirname(__file__))
prefixes = [scriptDir, '/usr/lib', '/usr/local/lib']

# Try to find the library using each of the available prefixes, in order
dll = None
searched = []
for prefix in prefixes:
    searchName = os.path.join(prefix, libName)
    try:
        dll = ctypes.cdll.LoadLibrary(searchName)
    except OSError:
        searched.append(searchName)

if dll is None:
   raise OSError('Cannot find library ' + libName + '. Searched the ' +
        'following paths: ' + '\n'.join(searched))

# Extract the warping function and set up its signature
warpfun = dll.cuda_image_warp
warpfun.argtypes = [
        ctypes.POINTER(ctypes.c_float), 
        ctypes.c_int, 
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
        ]
warpfun.restype = ctypes.c_int
