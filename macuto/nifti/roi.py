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

from ..strings import list_search


def drain_rois(img_data):
    """
    Retrieves the ROIs in img_data and returns a similar matrix with the ROIs
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


def create_rois_mask (roislist, filelist):
    """

    @param roislist:
    @param filelist:
    @return:
    """

    shape = nib.load(filelist[0]).shape
    mask  = np.zeros(shape)

    #create space for all features and read from subjects
    for roi in roislist:
        try:
            roif   = list_search('_' + roi + '.', filelist)[0]
            roivol = nib.load(roif).get_data()
            mask += roivol
        except exc:
            log.error(exc.message + '\n' + exc.argument)
            return 0


    return mask > 0
