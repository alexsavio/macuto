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

import h5py
import numpy as np
import nibabel as nib


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

    @param append=False
    True if you don't want to erase the content of the file
    if it already exists, False otherwise.

    @note:
    HDF5 open modes
    >>> f = h5py.File('filename.hdf5')       # opens, or creates if it doesn't exist
    >>> f = h5py.File('filename.hdf5','r')   # readonly
    >>> f = h5py.File('filename.hdf5','r+')  # read/write
    >>> f = h5py.File('filename.hdf5','w')   # new file overwriting any existing file
    >>> f = h5py.File('filename.hdf5','w-')  # new file only if one doesn't exist
    """
    mode = 'w'
    if append:
        mode = 'r+'

    f = h5py.File(fname, mode)

    h5img = f.create_group(h5path)
    h5img['data']   = spatial_img.get_data()
    h5img['extra']  = spatial_img.get_extra()
    h5img['affine'] = spatial_img.get_affine()

    hdr = spatial_img.get_header()
    for k in hdr.keys():
        h5img['data'].attrs[k] = hdr[k]

    f.close()


def hdfgroup_to_nifti1image(fname, h5path):
    """
    Returns a nibabel Nifti1Image from an HDF5 group datasets

    @param fname: string
    HDF5 file path

    @param h5path:
    HDF5 group path in fname

    @return: nibabel Nifti1Image
    """
    f = h5py.File(fname, 'r')

    h5img  = f[h5path]
    data   = h5img['data'].value
    extra  = h5img['extra'].value
    affine = h5img['affine'].value

    header = get_nifti1hdr_from_h5attrs(h5img['data'].attrs)

    f.close()

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
    for k in h5attrs.keys():
        hdr[str(k)] = np.array(h5attrs[k])

    return hdr

