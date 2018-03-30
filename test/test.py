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

# Load the image
im = nib.load(inPath)
data = im.get_data()

# Warp and add noise
dataWarp = cudaImageWarp.cudaImageWarp(data, A, interp='linear', std=50.0)

# Write the output
imOut = nib.Nifti1Image(dataWarp, im.affine)
nib.save(imOut, outPath)
