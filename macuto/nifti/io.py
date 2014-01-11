# coding=utf-8
#-------------------------------------------------------------------------------
#License GNU/GPL v3
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import os

import h5py
import numpy as np
import nibabel as nib
import logging as log


def save_nibabel(ofname, vol, affine, header=None):
    """
    Saves a volume into a Nifti (.nii.gz) file.

    Parameters
    ===========
    @param ofname: string
        File relative path and name
    @param vol: Numpy 3D or 4D array
        Volume with the data to be saved.
    @param affine: 4x4 Numpy array
        Array with the affine transform of the file.
    @param header: nibabel.nifti1.Nifti1Header, optional
        Header for the file, optional but recommended.
    """
    log.debug('Saving nifti file: ' + ofname)
    ni = nib.Nifti1Image(vol, affine, header)
    nib.save(ni, ofname)


def spatialimg_to_hdf(fname, spatial_img, h5path='/img', append=True):
    """
    Saves a Nifti1Image into an HDF5 file.

    @param fname: string
    Output HDF5 file path

    @param spatial_img: nibabel SpatialImage
    Image to be saved

    @param h5path: string
    HDF5 group path where the image data will be saved.
    Datasets will be created inside the given group path:
    'data', 'extra', 'affine', the header information will
    be set as attributes of the 'data' dataset.

    @param append: bool
    True if you don't want to erase the content of the file
    if it already exists, False otherwise.

    @note:
    HDF5 open modes
    >>> 'r' Readonly, file must exist
    >>> 'r+' Read/write, file must exist
    >>> 'w' Create file, truncate if exists
    >>> 'w-' Create file, fail if exists
    >>> 'a' Read/write if exists, create otherwise (default)

    """
    if not os.path.exists(fname):
        mode = 'w'
    else:
        if append:
            mode = 'a'

    with h5py.File(fname, mode) as f:

        try:
            h5img = f.create_group(h5path)
            h5img['data'] = spatial_img.get_data()
            h5img['affine'] = spatial_img.get_affine()

            if hasattr(h5img, 'get_extra'):
                h5img['extra'] = spatial_img.get_extra()

            hdr = spatial_img.get_header()
            for k in list(hdr.keys()):
                h5img['data'].attrs[k] = hdr[k]

        except ValueError as ve:
            log.error('Error creating group ' + h5path)
            print(str(ve))


def hdfgroup_to_nifti1image(fname, h5path):
    """
    Returns a nibabel Nifti1Image from a HDF5 group datasets

    @param fname: string
    HDF5 file path

    @param h5path:
    HDF5 group path in fname

    @return: nibabel Nifti1Image
    """
    with h5py.File(fname, 'r') as f: 

        h5img  = f[h5path]
        data   = h5img['data'][()]
        extra  = h5img['extra'][()]
        affine = h5img['affine'][()]

        header = get_nifti1hdr_from_h5attrs(h5img['data'].attrs)

    img = nib.Nifti1Image(data, affine, header=header, extra=extra)

    return img


def get_nifti1hdr_from_h5attrs(h5attrs):
    """
    Transforms an H5py Attributes set to a dict.
    Converts unicode string keys into standard strings
    and each value into a numpy array.

    @param h5attrs: H5py Attributes

    @return: dict
    """
    hdr = nib.Nifti1Header()
    for k in list(h5attrs.keys()):
        hdr[str(k)] = np.array(h5attrs[k])

    return hdr

