"""
Warp an image using scipy. This is a CPU-only implementation of cudaImageWarp.py. 

(c) Blaine Rister 2018
"""

import numpy as np
import scipy.ndimage as nd
from cudaImageWarp import __check_inputs
import pyCudaImageWarp

def push(im, A, interp='linear', shape=None, std=0.0, 
	winMin=-float('inf'), winMax=float('inf'), occZmin=0, occZmax=-1):
    """
        Reimplementation of push() in cudaImageWarp.py
    """

    # Check inputs
    shape = __check_inputs(im, A, shape)

    # Map from interpolation mode to spline order
    interpMap = {
        'nearest': 0,
        'linear': 1
    }

    # Affine warping
    im = nd.affine_transform(
        im, 
        A, 
        output_shape=shape, 
        order=interpMap[interp], 
        mode='constant',
        cval=0.0,
        prefilter=False
    )

    # Gaussian noise
    if std > 0.0:
        im += np.random.normal(scale=std)

    # Window
    im = np.maximum(np.minimum(im, winMin), winMax)
    win_width = winMax - winMin
    if win_width != float('inf') and win_width != -float('inf'):
        im /= win_width

    # Occlusion
    occLo = max(occZmin + 1, 1)
    occHi = min(occZmax + 1, shape[2])
    if occHi > occLo:
        im[:, :, occLo : occHi] = 0.0

    # Push all the output into a queue, for compatibility with the GPU version
    pyCudaImageWarp.q.put(im)

def pop():
    """
        Reimplementaiton of pop() in cudaImageWarp.py
    """
    return pyCudaImageWarp.q.get_nowait()
