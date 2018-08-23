import nibabel as nib 
import numpy as np

import sys

from pyCudaImageWarp import cudaImageWarp

# Usage: python test.py [input.nii.gz] [output.nii.gz]

inPath = sys.argv[1]
outPath = sys.argv[2]

# Warping matrix
A = np.array([[1, 0, 0, 30], 
              [0, 1, 0, 0],
              [0, 0, 2, 0]])
zMin = 100
zMax = zMin + 50

# Load the image
im = nib.load(inPath)
data = im.get_data()

# Warp, add noise and occlude
dataWarp = cudaImageWarp.warp(data, A, interp='linear', std=50.0, occZmin=zMin,
        occZmax=zMax)

# Write the output
imOut = nib.Nifti1Image(dataWarp, im.affine, header=im.header)
nib.save(imOut, outPath)

# Warp without noise
dataWarp = cudaImageWarp.warp(data, A, interp='linear', occZmin=zMin, 
        occZmax=zMax)

# Warp with the push/pop method, and ensure the results are the same
numIters = 2
for i in range(numIters):
        cudaImageWarp.push(data, A, interp='linear', occZmin=zMin, occZmax=zMax)
for i in range(numIters):
        dataWarp2 = cudaImageWarp.pop()

assert(np.all(np.equal(dataWarp, dataWarp2)))

