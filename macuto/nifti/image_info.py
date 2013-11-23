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

import nibabel as nib


def is_valid_coordinate(img, i, j, k):
    """
    @param img:
    @param i:
    @param j:
    @param k:
    @return:
    """
    imgX, imgY, imgZ = img.shape
    return (i >= 0 and i < imgX) and \
           (j >= 0 and j < imgY) and \
           (k >= 0 and k < imgZ)


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
    img1 = nib.load(fname1)
    img2 = nib.load(fname2)
    return img1.get_shape() == img2.get_shape()


def check_have_same_geometry (fname1, fname2):
    """
    @param fname1:
    @param fname2:
    @return:
    """
    if not have_same_geometry (fname1, fname2):
        err = 'Different shapes:' + fname1 + ' vs. ' + fname2
        raise IOError(err)