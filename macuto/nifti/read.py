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

log = logging.getLogger(__name__)


def get_nii_info(nii_file):
    """
    @param nii_file: string
    @return:
    hdr, aff
    """
    try:
        nibf = nib.load(nii_file)
        aff = nibf.get_affine()
        hdr = nibf.get_header()

    except Exception as exc:
        log.exception('Error reading file {0}.'.format(nii_file))

    return hdr, aff


def get_nii_data(nii_file):
    """
    @param nii_file: string
    @return:
    """
    try:
        nibf = nib.load(nii_file)
        vol = nibf.get_data()

    except Exception as exc:
        log.exception('Error on reading file {0}.'.format(nii_file))

    return vol


def load_nipy_img(nii_file):
    """

    :param nii_file: str
     Path to the nifti file

    :return: nipy.Image
    """
    if not os.path.exists(nii_file):
        raise FileNotFoundError(log, nii_file)

    try:
        return nipy.load_image(nii_file)
    except Exception as exc:
        log.exception('Reading file {0}.')


def get_masked_nii_data(nii_file, mask_file):
    """
    @param nii_file: string
    @param mask_file: string
    @return:
    vol[mask_indices], mask_indices, mask.shape
    """
    try:
        nibf = nib.load(nii_file)
        vol = nibf.get_data()

        mask = get_nii_data(mask_file)
        mask_indices = np.where(mask > 0)
    except Exception:
        log.exception('Reading file {0}.'.format(nii_file))

    return vol[mask_indices], mask_indices, mask.shape


def vector_to_volume(vector, mask_indices, mask_shape, dtype=None):
    """
    Transform a given vector to a volume. This is a reshape function for
    3D flattened and maybe masked vectors.

    :param vector: np.array

    :param mask_indices: np.array
    mask_indices = np.where(mask > 0)

    :param mask_shape: tuple

    :param dtype: return type
    If None, will get the type from vector

    :return:
    np.array
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

    Arguments:
    ----------
    nii_filelist:  list of strings
    List of absolute file paths to nifti files. All nifti files must have the
    same shape.

    smoothmm: int
    Integer indicating the size of the FWHM Gaussian smoothing kernel you would
    like for smoothing the volume before flattening it.
    Need FSL and nipype.
    See smooth_volume() source code.

    outdtype: dtype
    Type of the elements of the array, if not set will obtain the dtype from
    the first nifti file.

    Returns:
    --------
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
    """
    From the list of absolute paths to nifti files, creates a Numpy array
    with the masked data.

    Arguments:
    ----------
    @param nii_filelist: list of strings
    List of absolute file paths to nifti files. All nifti files must have the
    same shape.

    @param mask_file: string
    Path to a Nifti mask file.
    Should be the same shape as the files in nii_filelist.

    @param outdtype: dtype
    Type of the elements of the array, if not set will obtain the dtype from
    the first nifti file.

    Returns:
    --------
    @return
    outmat: Numpy array with shape N x mask_voxels
            containing the N files as flat vectors with the data within
            the mask file.

    mask_indices: Tuple with the 3D spatial indices of the masking voxels, for
    reshaping with vol_shape and remapping.

    vol_shape: Tuple with shape of the volumes, for reshaping.

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
