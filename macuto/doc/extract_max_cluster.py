#!/usr/bin/python

import nibabel as nib
import numpy as np

imf = 'T1_brain_cluster_mask.nii.gz'
img = nib.load(imf)
vol = img.get_data()
idx = vol == np.max(vol)
vol = vol * 0
vol[idx] = 1
new_image = nib.Nifti1Image(vol, img.get_affine())
nib.save(new_image, imf + '_maxcluster.nii.gz')
