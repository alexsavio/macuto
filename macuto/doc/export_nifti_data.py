#!/usr/bin/python

import os
import re
import shelve
import tables as tabs
import numpy as np
import nibabel as nib
import scipy.io as sio

#-------------------------------------------------------------------------------
def get_extension (fpath, check_if_exists=False):
    if check_if_exists:
        if not os.path.exists (fpath):
            err = 'File not found: ' + fpath
            raise IOError(err)
    try:
        s = os.path.splitext(fpath)
        return s[-1]
    except:
        print( "Unexpected error: ", sys.exc_info()[0] )
        raise

#-------------------------------------------------------------------------------
def add_extension_if_needed (fpath, ext, check_if_exists=False):
    if fpath.find(ext) < 0:
        fpath += ext
    if check_if_exists:
        if not os.path.exists (fpath):
            err = 'File not found: ' + fpath
            raise IOError(err)
    return fpath

#-------------------------------------------------------------------------------
def get_nii_data(niipath):
    try:
        nibf = nib.load(niipath)
        vol  = nibf.get_data()
        aff  = nibf.get_affine()
        hdr  = nibf.get_header()

        return vol, hdr, aff

    except:
        print ('get_nii_data: Error on reading file ' + niipath)

#-------------------------------------------------------------------------------
def shelve_varlist (outfpath, varnames, varlist):
    '''
    before calling this function, create a varlist this way:

    shelfvars = []
    for v in varnames:
        shelfvars.append(eval(v))
    '''

    mashelf = shelve.open(outfpath, 'n')

    for i in np.arange(len(varnames)):
       try:
           mashelf[varnames[i]] = varlist[i]
       except:
           print('ERROR shelving: {0}'.format(varnames[i]))
           print sys.exc_info()

    mashelf.close()

   #to_restore
   #my_shelf = shelve.open(filename)
   #for key in my_shelf:
   #   globals()[key]=my_shelf[key]
   #my_shelf.close()

#-------------------------------------------------------------------------------
def save_varlist_to_mat (outfpath, varnames, varlist):
    '''
    '''
    mdict = {}
    try:
        for i,var in enumerate(varlist):
            mdict[varnames[i]] = var

        sio.savemat(outfpath, mdict, format='4')
    except:
        print('ERROR saving to .mat: {0}'.format(varnames[i]))
        print sys.exc_info()

#-------------------------------------------------------------------------------
def save_varlist_to_hdf5 (outfpath, varnames, varlist):
    '''
    '''
    h5file = tabs.open_file(outfpath, mode = "w", title = os.path.basename(outfpath))
    root = h5file.root

    try:
        for i,var in enumerate(varlist):
            h5file.createArray(root, varnames[i], var)

    except:
        print('ERROR saving to .hdf5: {0}'.format(varnames[i]))
        print sys.exc_info()

    h5file.close()

#-------------------------------------------------------------------------------
def find (lst, regex):
    o = []
    for i in lst:
        if re.search (regex, i):
            o.append(i)
    return o

#-------------------------------------------------------------------------------
def smooth_volume(imf, smoothmm):
    from nipype.interfaces.fsl.maths import IsotropicSmooth

    if smoothmm > 0:
        omf = imf + '_smooth' + str(smoothmm) + 'mm.nii.gz'

        isosmooth = IsotropicSmooth()
        isosmooth.inputs.in_file  = imf
        isosmooth.inputs.fwhm     = smoothmm
        isosmooth.inputs.out_file = omf
        isosmooth.run()

        data = nib.load(omf).get_data()
        os.remove(omf)
    else:
        data = nib.load(imf).get_data()
    return data

#-------------------------------------------------------------------------------
def niftilist_to_matrix (niftilist, maskfile, output_file, outdtype=np.float16):

    mask, hdr, aff = get_nii_data(maskfile)
    mask_indices = np.where(mask > 0)

    outmat = np.zeros((len(niftilist), np.count_nonzero(mask)), dtype=outdtype)

    for i,vf in enumerate(niftilist):
        vol, hdr, aff = get_nii_data(vf)
        outmat[i, :] = vol[mask_indices]

    return outmat, mask_indices, mask.shape

#-------------------------------------------------------------------------------
#start variables
#jacs
nifti_dir   = '/home/alexandre/Data/oasis/tbm_features/jacs'
output_file = '/home/alexandre/Data/oasis/tbm_features/jacs.hdf5'

#modulatedgm
nifti_dir   = '/home/alexandre/Data/oasis/tbm_features/modulatedgm'
output_file = '/home/alexandre/Data/oasis/tbm_features/modulatedgm.hdf5'

mask_file   = '/home/alexandre/Data/oasis/tbm_features/MNI152_T1_1mm_brain_mask_dil.nii.gz'
labelsf     = '/home/alexandre/Data/oasis/tbm_features/labels'
outdtype    = np.float32

#setup output file extension
ext = get_extension(output_file)
if ext != '.pyshelf' and ext != '.mat' and ext != '.hdf5':
    output_file = add_extension_if_needed (output_file, '.pyshelf')
    ext = get_extension(output_file)

print ('Loading data')
#create nifitlist
niftilist = os.listdir(nifti_dir)
niftilist.sort()
niftilist = find(niftilist, 'nii*')
niftilist = [os.path.join(nifti_dir, nifti) for nifti in niftilist]

#load and mask data
outmat, indices, vol_shape = niftilist_to_matrix (niftilist, mask_file, output_file, outdtype)

#read labels
labels = np.loadtxt(labelsf, dtype=int)

print ('Saving ' + output_file)

varnames = ['data', 'labels', 'mask_indices', 'vol_shape']
varlist  = [outmat, labels, indices, vol_shape]

if ext == '.pyshelf':
    shelve_varlist (output_file, varnames, varlist)

elif ext == '.mat':
    save_varlist_to_mat (output_file, varnames, varlist)

elif ext == '.hdf5':
    save_varlist_to_hdf5 (output_file, varnames, varlist)

