"""
Test the data augmentation in augment3d.py

Usage: python test_augment3d.py [input.nii.gz]
"""

import sys
import numpy as np
import nibabel as nib

from pyCudaImageWarp import augment3d

inPath = sys.argv[1]

# Load the image
im = nib.load(inPath)
data = im.get_data()

# Test the augmenter with each transform
identity = augment3d.cuda_affine_augment3d([data])[0][0]
nib.save(nib.Nifti1Image(identity, im.affine, header=im.header), 'identity.nii.gz')

rotate = augment3d.cuda_affine_augment3d([data], rotMax=(90, 90, 90))[0][0]
nib.save(nib.Nifti1Image(rotate, im.affine, header=im.header), 'rotate.nii.gz')

reflect = augment3d.cuda_affine_augment3d([data], pReflect=(0, 0, 1))[0][0]
nib.save(nib.Nifti1Image(reflect, im.affine, header=im.header), 'reflect.nii.gz')

shear = augment3d.cuda_affine_augment3d([data], shearMax=(2,0,0))[0][0]
nib.save(nib.Nifti1Image(shear, im.affine, header=im.header), 'shear.nii.gz')

translate = augment3d.cuda_affine_augment3d([data], transMax=(30,30,30))[0][0]
nib.save(nib.Nifti1Image(translate, im.affine, header=im.header), 'translate.nii.gz')

other = augment3d.cuda_affine_augment3d([data], otherScale=0.33)[0][0]
nib.save(nib.Nifti1Image(other, im.affine, header=im.header), 'other.nii.gz')

crop = augment3d.cuda_affine_augment3d([data], shapeList=[(100, 100, 100)])[0][0]
nib.save(nib.Nifti1Image(crop, im.affine, header=im.header), 'crop.nii.gz')

noise = augment3d.cuda_affine_augment3d([data], noiseLevel=50)[0][0]
nib.save(nib.Nifti1Image(noise, im.affine, header=im.header), 'noise.nii.gz')

window = augment3d.cuda_affine_augment3d([data], windowMin=(0,) * 2, 
	windowMax=(150,) * 2)[0][0]
nib.save(nib.Nifti1Image(window, im.affine, header=im.header), 'window.nii.gz')

occlude = augment3d.cuda_affine_augment3d([data], occludeProb=1.0)[0][0]
nib.save(nib.Nifti1Image(occlude, im.affine, header=im.header), 'occlude.nii.gz')

# Test the Scipy backup implementation
cpu_rotate = augment3d.cuda_affine_augment3d([data], rotMax=(90, 90, 90), api='scipy')[0][0]
nib.save(nib.Nifti1Image(rotate, im.affine, header=im.header), 'scipy_rotate.nii.gz')
