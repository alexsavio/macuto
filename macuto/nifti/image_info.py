# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import nibabel as nib


def is_valid_coordinate(img, i, j, k):
    """
    @param img:
    @param i:
    @param j:
    @param k:
    @return:
    """
    imgx, imgy, imgz = img.shape
    return (i >= 0 and i < imgx) and \
           (j >= 0 and j < imgy) and \
           (k >= 0 and k < imgz)


def are_compatible_imgs(one_img, another_img):
    """
    Returns true if one_img and another_img have the same shape, false
    otherwise.

    @param one_img:
    @param another_img:
    @return:
    """
    return have_same_shapes(one_img, another_img)


def have_same_shapes(array1, array2):
    """
    Returns true if array1 and array2 have the same shapes, false
    otherwise.

    @param array1:
    @param array2:
    @return:
    """
    return array1.shape == array2.shape


def have_same_geometry(fname1, fname2):
    """
    @param fname1: string
    File path of an image

    @param fname2: string
    File path of an image

    @return: bool
    True if both have the same geometry
    """
    img1shape = nib.load(fname1).get_shape()
    img2shape = nib.load(fname2).get_shape()
    return img1shape == img2shape


def have_same_spatial_geometry(fname1, fname2):
    """
    @param fname1: string
    File path of an image

    @param fname2: string
    File path of an image

    @return: bool
    True if both have the same geometry
    """
    img1shape = nib.load(fname1).get_shape()
    img2shape = nib.load(fname2).get_shape()
    return img1shape[:3] == img2shape[:3]


def check_have_same_geometry(fname1, fname2):
    """
    @param fname1:
    @param fname2:
    @return:
    """
    if not have_same_geometry(fname1, fname2):
        err = 'Different shapes:' + fname1 + ' vs. ' + fname2
        raise ArithmeticError(err)


def check_have_same_spatial_geometry(fname1, fname2):
    """
    @param fname1:
    @param fname2:
    @return:
    """
    if not have_same_spatial_geometry(fname1, fname2):
        err = 'Different shapes:' + fname1 + ' vs. ' + fname2
        raise ArithmeticError(err)


def get_sampling_interval(func_img):
    """
    Extracts the supposed sampling interval (TR) from the nifti file header.

    @param func_img: a NiBabel SpatialImage

    @return: float
    The TR value from the image header
    """
    return func_img.get_header().get_zooms()[-1]