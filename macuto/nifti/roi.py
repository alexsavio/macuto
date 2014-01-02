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


import numpy as np
import nibabel as nib
import logging as log
import scipy.ndimage as scn
from macuto.nifti.image_info import check_have_same_spatial_geometry

from ..strings import list_search


def drain_rois(img_data):
    """
    Find all the ROIs in img_data and returns a similar volume with the ROIs
    emptied, keeping only their border voxels.

    This is useful for DTI tractography.

    @param img_data: numpy array

    @return:
    an array of same shape as img_data
    """

    out = np.zeros(img_data.shape, dtype=img_data.dtype)

    if img_data.ndim == 2:
        kernel = np.ones([3, 3], dtype=int)
    elif img_data.ndim == 3:
        kernel = np.ones([3, 3, 3], dtype=int)
    elif img_data.ndim == 4:
        kernel = np.ones([3, 3, 3, 3], dtype=int)

    vals = np.unique(img_data)
    vals = vals[vals != 0]

    for i in vals:
        roi  = img_data == i
        hits = scn.binary_hit_or_miss(roi, kernel)
        roi[hits] = 0
        out[roi > 0] = i

    return out


def create_rois_mask(roislist, filelist):
    """
    Looks for the files in filelist containing the names
    in roislist, these files will be opened, binarised
    and merged in one mask.

    @param roislist: list of strings
    Names of the ROIs, which will have to be in the
    names of the files in filelist.

    @param filelist: list of strings
    List of paths to the volume files containing the ROIs.

    @return: ndarray
    Mask volume
    """
    roifiles = []

    for roi in roislist:
        try:
            roifiles.apend(list_search(roi, filelist)[0])
        except Exception as exc:
            log.error(exc.message + '\n' + exc.argument)
            return 0

    return create_mask_from(roifiles)


def create_mask_from(filelist):
    """
    Creates a binarised mask with the files in
    filelist.

    @param filelist: list of strings
    List of paths to the volume files containing the ROIs.

    @return: ndarray of int
    Mask volume
    """
    shape = nib.load(filelist[0]).shape
    mask  = np.zeros(shape)

    #create space for all features and read from subjects
    for volf in filelist:
        try:
            roivol = nib.load(volf).get_data()
            mask += roivol
        except Exception as exc:
            log.error(exc.message + '\n' + exc.argument)
            return 0

    return (mask > 0).astype(int)


def get_roilist_from_atlas(atlas):
    """
    Extract unique values from the atlas and returns them as an ordered list.

    @param atlas: ndarray
    Volume defining different ROIs.

    @return: ndarray
    An 1D array of roi values from atlas volume.

    Note
    ----
    The roi with value 0 will be considered background so will be removed.
    """
    rois = np.unique(atlas)
    rois = rois[np.nonzero(rois)]
    rois.sort()

    return rois


def extract_timeseries(tsvol, roivol):
    """
    Partitions the timeseries in tsvol according to the
    ROIs in roivol.

    @param tsvol: ndarray
    4D timeseries volume

    @param roivol: ndarray
    3D ROIs volume

    @return: dict
    A dict with the timeseries as items and
    keys as the ROIs voxel values.
    """
    assert(tsvol.ndim == 4)
    assert(tsvol.shape[0:2] == roivol.shape)

    rois = get_roilist_from_atlas(roivol)

    tsmat = tsvol.reshape((np.prod(tsvol.shape[0:2]), tsvol.shape[3]))

    ts_dict = {}
    for r in rois:
        ts_dict[r] = tsmat[roivol == r, :]

    return ts_dict