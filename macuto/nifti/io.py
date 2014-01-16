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


def spatialimg_to_hdfgroup(h5group, spatial_img):
    """
    Saves a Nifti1Image into an HDF5 group.

    @param h5group: h5py Group
    Output HDF5 file path

    @param spatial_img: nibabel SpatialImage
    Image to be saved

    @param h5path: string
    HDF5 group path where the image data will be saved.
    Datasets will be created inside the given group path:
    'data', 'extra', 'affine', the header information will
    be set as attributes of the 'data' dataset.

    """
    try:
        h5group['data'] = spatial_img.get_data()
        h5group['affine'] = spatial_img.get_affine()

        if hasattr(h5group, 'get_extra'):
            h5group['extra'] = spatial_img.get_extra()

        hdr = spatial_img.get_header()
        for k in list(hdr.keys()):
            h5group['data'].attrs[k] = hdr[k]

    except ValueError as ve:
        log.error('Error creating group ' + h5group.name)
        print(str(ve))


def spatialimg_to_hdfpath(fname, spatial_img, h5path='/img', append=True):
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
    mode = 'w'
    if os.path.exists(fname):
        if append:
            mode = 'a'

    with h5py.File(fname, mode) as f:
        try:
            h5img = f.create_group(h5path)
            spatialimg_to_hdfgroup(h5img, spatial_img)

        except ValueError as ve:
            log.error('Error creating group ' + h5path)
            print(str(ve))


def hdfpath_to_nifti1image(fname, h5path):
    """
    Returns a nibabel Nifti1Image from a HDF5 group datasets

    @param fname: string
    HDF5 file path

    @param h5path:
    HDF5 group path in fname

    @return: nibabel Nifti1Image
    """
    with h5py.File(fname, 'r') as f: 
        return hdfgroup_to_nifti1image(f[h5path])


def hdfgroup_to_nifti1image(h5group):
    """
    Returns a nibabel Nifti1Image from a HDF5 group datasets

    @param fname: h5py.Group
    HDF5 group

    @return: nibabel Nifti1Image
    """
    try:
        data   = h5group['data'][:]
        affine = h5group['affine'][:]

        extra = None
        if 'extra' in h5group:
            extra = h5group['extra'][:]

        header = get_nifti1hdr_from_h5attrs(h5group['data'].attrs)

    except KeyError as ke:
        log.error('Could not read Nifti1Image datasets from ' + h5group.name)
        print(str(ke))

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


def all_childnodes_to_nifti1img(h5group):
    """
    Returns in a list all images found under h5group.

    @param h5group: h5py.Group
    HDF group

    @return: list of nifti1Image
    """
    child_nodes = []
    def append_parent_if_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            if name.split('/')[-1] == 'data':
                child_nodes.append(obj.parent)

    vols = []
    h5group.visititems(append_parent_if_dataset)
    for c in child_nodes:
        vols.append(hdfgroup_to_nifti1image(c))

    return vols