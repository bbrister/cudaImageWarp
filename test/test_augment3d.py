"""
Test the data augmentation in augment3d.py

Usage: python test_augment3d.py [input.nii.gz]
"""

import sys
import numpy as np
import nibabel as nib

from pyCudaImageWarp import augment3d

def apply_and_write_output(im, xform, name, api='cuda', oob=0):
    out = augment3d.apply_xforms_image([xform], [im.get_data()], api=api,
        oob_image_val=oob)[0][0]
    nib.save(nib.Nifti1Image(out, im.affine, header=im.header), name) 

inPath = sys.argv[1]

# Load the image
im = nib.load(inPath)
data = im.get_data()

# Test the augmenter with each transform
identity = augment3d.get_xform(data)
apply_and_write_output(im, identity, 'identity.nii.gz')

rotate = augment3d.get_xform(data, rotMax=(90, 90, 90))
apply_and_write_output(im, rotate, 'rotate.nii.gz')

reflect = augment3d.get_xform(data, pReflect=(0, 0, 1))
apply_and_write_output(im, reflect, 'reflect.nii.gz')

shear = augment3d.get_xform(data, shearMax=(2,0,0))
apply_and_write_output(im, shear, 'shear.nii.gz')

translate = augment3d.get_xform(data, transMax=(30,30,30))
apply_and_write_output(im, translate, 'translate.nii.gz')

other = augment3d.get_xform(data, otherScale=0.33)
apply_and_write_output(im, other, 'other.nii.gz')

crop = augment3d.get_xform(data, shape=(100, 100, 100))
apply_and_write_output(im, crop, 'crop.nii.gz')

noise = augment3d.get_xform(data, noiseLevel=[50])
apply_and_write_output(im, noise, 'noise.nii.gz')

window = augment3d.get_xform(data, windowMin=np.array([[0],[0]]), 
	windowMax=np.array([[150],[150]]))
apply_and_write_output(im, window, 'window.nii.gz')

occlude = augment3d.get_xform(data, occludeProb=1.0)
apply_and_write_output(im, occlude, 'occlude.nii.gz', oob=100)

# Test the Scipy backup implementation
apply_and_write_output(im, rotate, 'scipy_rotate.nii.gz', api='scipy')
