# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import os
import nipy
import numpy as np
import nibabel as nib
import logging

from .process import are_compatible_imgs
from ..exceptions import FileNotFound, NiftiFilesNotCompatible

log = logging.getLogger(__name__)


def get_nii_info(nii_file):
    """Return the header and affine matrix from a Nifti file.

    Parameters
    ----------
    param nii_file: str
        Nifti file path

    Returns
    -------
    hdr, aff
    """
    if not os.path.exists(nii_file):
        raise FileNotFound(nii_file)

    try:
        nibf = nib.load(nii_file)

        return nibf.get_header(), nibf.get_affine()
    except Exception as exc:
        log.exception('Error reading file {0}.'.format(nii_file))


def get_nii_data(nii_file):
    """Return the voxel matrix of the Nifti file

    Parameters
    ----------
    param nii_file: str
        Nifti file path

    Returns
    -------
    array_like
    """
    if not os.path.exists(nii_file):
        raise FileNotFound(nii_file)

    try:
        nibf = nib.load(nii_file)
        vol = nibf.get_data()
        return vol
    except Exception as exc:
        log.exception('Error on reading file {0}.'.format(nii_file))


def load_nipy_img(nii_file):
    """Read a Nifti file and return as nipy.Image

    Parameters
    ----------
    param nii_file: str
        Nifti file path

    Returns
    -------
    nipy.Image
    """
    if not os.path.exists(nii_file):
        raise FileNotFound(nii_file)

    try:
        return nipy.load_image(nii_file)
    except Exception as exc:
        log.exception('Reading file {0}.')


def get_masked_nii_data(nii_file, mask_file):
    """Read a Nifti file nii_file and a mask Nifti file.
    Returns the voxels in nii_file that are within the mask, the mask indices
    and the mask shape.

    Parameters
    ----------
    param nii_file: str
        Nifti file path

    param mask_file: str
        Nifti mask file path

    Returns
    -------
    vol[mask_indices], mask_indices, mask.shape

    Note
    ----
    nii_file and mask_file must have the same shape.

    Raises
    ------
    FileNotFound, NiftiFilesNotCompatible
    """
    if not os.path.exists(nii_file):
        raise FileNotFound(nii_file)

    if not os.path.exists(mask_file):
        raise FileNotFound(mask_file)

    if not are_compatible_imgs(nii_file, mask_file):
        raise NiftiFilesNotCompatible(nii_file, mask_file)

    try:
        nibf = nib.load(nii_file)
        vol = nibf.get_data()

        mask = get_nii_data(mask_file)
        mask_indices = np.where(mask > 0)
        return vol[mask_indices], mask_indices, mask.shape

    except Exception:
        log.exception('Reading file {0}.'.format(nii_file))


def vector_to_volume(vector, mask_indices, mask_shape, dtype=None):
    """Transform a given vector to a volume. This is a reshape function for
    3D flattened and maybe masked vectors.

    Parameters
    ----------
    vector: np.array

    mask_indices: tuple of ndarrays
        mask_indices = np.where(mask > 0)

    mask_shape: tuple

    dtype: return type
        If None, will get the type from vector

    Returns
    -------
    np.ndarray
    """
    if dtype is None:
        dtype = vector.dtype

    try:
        volume = np.zeros(mask_shape, dtype=dtype)
        volume[mask_indices] = vector
        return volume
    except Exception as exc:
        log.exception('Error on transforming vector to volume.')


def niftilist_to_array(nii_filelist, outdtype=None):
    """
    From the list of absolute paths to nifti files, creates a Numpy array
    with the data.

    Parameters
    ----------
    nii_filelist:  list of str
        List of absolute file paths to nifti files. All nifti files must have
        the same shape.

    smoothmm: int
        Integer indicating the size of the FWHM Gaussian smoothing kernel you 
        would like for smoothing the volume before flattening it.
        Need FSL and nipype.
        See smooth_volume() source code.

    outdtype: dtype
        Type of the elements of the array, if not set will obtain the dtype from
        the first nifti file.

    Returns
    -------
    outmat: Numpy array with shape N x prod(vol.shape)
            containing the N files as flat vectors.

    vol_shape: Tuple with shape of the volumes, for reshaping.

    """
    try:
        vol = get_nii_data(nii_filelist[0])
    except IndexError as ie:
        log.exception('nii_filelist should not be empty.')

    if not outdtype:
        outdtype = vol.dtype

    outmat = np.zeros((len(nii_filelist), np.prod(vol.shape)), dtype=outdtype)

    try:
        for i, vf in enumerate(nii_filelist):
            vol = get_nii_data(vf)
            outmat[i, :] = vol.flatten()
    except:
        log.exception('Error on reading file {0}.'.format(vf))

    return outmat, vol.shape


def niftilist_mask_to_array(nii_filelist, mask_file=None, outdtype=None):
    """From the list of absolute paths to nifti files, creates a Numpy array
    with the masked data.

    Parameters
    ----------
    nii_filelist: list of str
        List of absolute file paths to nifti files. All nifti files must have 
        the same shape.

    mask_file: str
        Path to a Nifti mask file.
        Should be the same shape as the files in nii_filelist.

    outdtype: dtype
        Type of the elements of the array, if not set will obtain the dtype from
        the first nifti file.

    Returns
    -------
    outmat: 
        Numpy array with shape N x mask_voxels containing the N files as flat 
        vectors with the data within the mask file.

    mask_indices: 
        Tuple with the 3D spatial indices of the masking voxels, for reshaping 
        with vol_shape and remapping.

    vol_shape: 
        Tuple with shape of the volumes, for reshaping.

    """
    vol = get_nii_data(nii_filelist[0])
    if not outdtype:
        outdtype = vol.dtype

    mask = get_nii_data(mask_file)
    mask_indices = np.where(mask > 0)

    outmat = np.zeros((len(nii_filelist), np.count_nonzero(mask)),
                      dtype=outdtype)

    try:
        for i, vf in enumerate(nii_filelist):
            vol = get_nii_data(vf)
            outmat[i, :] = vol[mask_indices]
    except Exception as exc:
        log.exception('Error on reading file {0}.'.format(vf))

    return outmat, mask_indices, mask.shape
