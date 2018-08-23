
"""
Python module initializer for CUDA image warper.

(c) Blaine Rister 2018
"""

import os
import ctypes

# Initialize a queue for recording inputs
try:
    import queue
    q = queue.queue()
except:
    import Queue
    q = Queue.Queue()

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

# Extract the single-image warping function
warpfun = dll.cuda_image_warp
warpfun.argtypes = [
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int, 
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int, 
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int
    ]
warpfun.restype = ctypes.c_int

# Extract the warp push function
pushfun = dll.cuda_image_warp_push
pushfun.argtypes = [
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int, 
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int, 
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int
    ]
pushfun.restype = ctypes.c_int

# Extract the warping end function
popfun = dll.cuda_image_warp_pop
popfun.argTypes = [
    ctypes.POINTER(ctypes.c_float)
    ]
popfun.restype = ctypes.c_int
