"""
Warp an image using scipy. This is a CPU-only implementation of cudaImageWarp.py. 

(c) Blaine Rister 2018
"""

import numpy as np
import scipy.ndimage as nd
import pyCudaImageWarp
import concurrent
import os

from .cudaImageWarp import __check_inputs

# Place to store worker threads
threadPool = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

def push(im, A, interp='linear', shape=None, std=0.0, 
        winMin=-float('inf'), winMax=float('inf'), occZmin=0, occZmax=-1, 
        oob=0.0, device=None):
    """
        Reimplementation of push() in cudaImageWarp.py
    """

    # Check inputs
    shape, device = __check_inputs(im, A, shape, device)

    # Map from interpolation mode to spline order
    interpMap = {
        'nearest': 0,
        'linear': 1
    }


    # Push the threadpool future into a queue
    pyCudaImageWarp.q.put(threadPool.submit(
        __warp_im__,
        im,
        A,
        interpMap[interp],
        shape,
        std,
        winMin,
        winMax,
        occZmin,
        occZmax,
        oob
    ))


def __warp_im__(im=None, A=None, interpCode=None, shape=None, std=None,
        winMin=None, winMax=None, occZmin=None, occZmax=None, oob=None):
    """
        Main function for image processing. Called in the thread pool.
    """
       
    # Affine warping
    im = nd.affine_transform(
        im, 
        A, 
        output_shape=shape, 
        order=interpCode, 
        mode='constant',
        cval=oob,
        prefilter=False
    )

    # Gaussian noise
    if std > 0.0:
        im += np.random.normal(scale=std)

    # Window
    im = np.maximum(np.minimum(im, winMax), winMin)
    win_width = winMax - winMin
    if win_width != float('inf') and win_width != -float('inf'):
        im -= winMin
        im /= win_width

    # Occlusion
    occLo = max(occZmin + 1, 1)
    occHi = min(occZmax + 1, shape[2])
    if occHi > occLo:
        im[:, :, occLo : occHi] = 0.0

    return im

def pop():
    """
        Reimplementaiton of pop() in cudaImageWarp.py
    """
    return pyCudaImageWarp.q.get_nowait().result(timeout=30)

def warp(im, A, interp='linear', shape=None, std=0.0, 
        winMin=-float('inf'), winMax=float('inf'), occZmin=0, occZmax=-1, oob=0,
        device=None):
    """
        Does a push() then pop()
    """
    push(im, A, interp=interp, shape=shape, std=std, winMin=winMin, winMax=winMax, 
        occZmin=occZmin, occZmax=occZmax, oob=oob, device=device)
    return pop()
