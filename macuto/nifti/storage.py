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

import h5py
import numpy as np
import nibabel as nib
import logging

import nipy.core.image as niim
from nipy import save_image

log = logging.getLogger(__name__)


def save_niigz(file_path, vol, affine=None, header=None):
    """Saves a volume into a Nifti (.nii.gz) file.

    Parameters
    ----------
    vol: Numpy 3D or 4D array
        Volume with the data to be saved.
    file_path: string
        Output file name path
    affine: 4x4 Numpy array
        Array with the affine transform of the file.
    header: nibabel.nifti1.Nifti1Header, optional
        Header for the file, optional but recommended.

    Note
    ----
        affine and header only work for numpy volumes.

    """
    if isinstance(vol, np.ndarray):
        log.debug('Saving numpy nifti file: ' + file_path)
        ni = nib.Nifti1Image(vol, affine, header)
        nib.save(ni, file_path)

    elif isinstance(vol, nib.Nifti1Image):
        log.debug('Saving nibabel nifti file: ' + file_path)
        nib.save(vol, file_path)

    elif isinstance(vol, niim.Image):
        log.debug('Saving nibabel nifti file: ' + file_path)
        save_image(vol, file_path)


def spatialimg_to_hdfgroup(h5group, spatial_img):
    """Saves a Nifti1Image into an HDF5 group.

    Parameters
    ----------
    h5group: h5py Group
        Output HDF5 file path

    spatial_img: nibabel SpatialImage
        Image to be saved

    h5path: str
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
        log.error(ve)
        raise


def spatialimg_to_hdfpath(file_path, spatial_img, h5path=None, append=True):
    """Saves a Nifti1Image into an HDF5 file.

    Parameters
    ----------
    file_path: string
        Output HDF5 file path

    spatial_img: nibabel SpatialImage
        Image to be saved

    h5path: string
        HDF5 group path where the image data will be saved.
        Datasets will be created inside the given group path:
        'data', 'extra', 'affine', the header information will
        be set as attributes of the 'data' dataset.
        Default: '/img'

    append: bool
        True if you don't want to erase the content of the file
        if it already exists, False otherwise.

    Note
    ----
    HDF5 open modes
    >>> 'r' Readonly, file must exist
    >>> 'r+' Read/write, file must exist
    >>> 'w' Create file, truncate if exists
    >>> 'w-' Create file, fail if exists
    >>> 'a' Read/write if exists, create otherwise (default)

    """
    if h5path is None:
        h5path = '/img'

    mode = 'w'
    if os.path.exists(file_path):
        if append:
            mode = 'a'

    with h5py.File(file_path, mode) as f:
        try:
            h5img = f.create_group(h5path)
            spatialimg_to_hdfgroup(h5img, spatial_img)

        except ValueError as ve:
            log.error('Error creating group ' + h5path)
            log.error(ve)
            raise


def hdfpath_to_nifti1image(file_path, h5path):
    """Returns a nibabel Nifti1Image from a HDF5 group datasets

    Parameters
    ----------
    file_path: string
        HDF5 file path

    h5path:
        HDF5 group path in file_path

    Returns
    -------
        nibabel Nifti1Image
    """
    with h5py.File(file_path, 'r') as f:
        return hdfgroup_to_nifti1image(f[h5path])


def hdfgroup_to_nifti1image(h5group):
    """Returns a nibabel Nifti1Image from a HDF5 group datasets

    Parameters
    ----------
    h5group: h5py.Group
        HDF5 group

    Returns
    -------
    nibabel Nifti1Image
    """
    try:
        data   = h5group['data'][:]
        affine = h5group['affine'][:]

        extra = None
        if 'extra' in h5group:
            extra = h5group['extra'][:]

        header = get_nifti1hdr_from_h5attrs(h5group['data'].attrs)

        img = nib.Nifti1Image(data, affine, header=header, extra=extra)

        return img

    except KeyError as ke:
        log.error('Could not read Nifti1Image datasets from ' + h5group.name)
        log.error(str(ke))
        raise


def get_nifti1hdr_from_h5attrs(h5attrs):
    """Transforms an H5py Attributes set to a dict.
    Converts unicode string keys into standard strings
    and each value into a numpy array.

    Parameters
    ----------
    h5attrs: H5py Attributes

    Returns
    --------
    dict
    """
    hdr = nib.Nifti1Header()
    for k in list(h5attrs.keys()):
        hdr[str(k)] = np.array(h5attrs[k])

    return hdr


def all_childnodes_to_nifti1img(h5group):
    """Returns in a list all images found under h5group.

    Parameters
    ----------
    h5group: h5py.Group
        HDF group

    Returns
    -------
    list of nifti1Image
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


def insert_volumes_in_one_dataset(file_path, h5path, file_list, newshape=None,
                                  concat_axis=0, dtype=None, append=True):
    """Inserts all given nifti files from file_list into one dataset in fname.
    This will not check if the dimensionality of all files match.

    Parameters
    ----------
    file_path: string
        HDF5 file path

    h5path: string

    file_list: list of strings

    newshape: tuple or lambda function
        If None, it will not reshape the images.
        If a lambda function, this lambda will receive only the shape array.
        e.g., newshape = lambda x: (np.prod(x[0:3]), x[3])
        If a tuple, it will try to reshape all the images with the same shape.
        It must work for all the images in file_list.

    concat_axis: int
        Axis of concatenation after reshaping

    dtype: data type
    Dataset data type
    If not set, will use the type of the first file.

    append: bool

    Raises
    ------
    ValueError if concat_axis is bigger than data dimensionality.

    Note
    ----
    For now, this only works if the dataset ends up being a 2D matrix.
    I haven't tested for multi-dimensionality concatenations.
    """

    def isalambda(v):
        return isinstance(v, type(lambda: None)) and v.__name__ == '<lambda>'

    mode = 'w'
    if os.path.exists(file_path):
        if append:
            mode = 'a'

    #loading the metadata into spatialimages
    imgs = [nib.load(vol) for vol in file_list]

    #getting the shapes of all volumes
    shapes = [np.array(img.get_shape()) for img in imgs]

    #getting the reshaped shapes
    if newshape is not None:
        if isalambda(newshape):
            nushapes = np.array([newshape(shape) for shape in shapes])
        else:
            nushapes = np.array([newshape for shape in shapes])

    #checking if concat_axis is available in this new shapes
    for nushape in nushapes:
        assert(len(nushape) - 1 < concat_axis)

    #calculate the shape of the new dataset
    n_dims = nushapes.shape[1]
    ds_shape = np.zeros(n_dims, dtype=np.int)
    for a in list(range(n_dims)):
        if a == concat_axis:
            ds_shape[a] = np.sum(nushapes[:, concat_axis])
        else:
            ds_shape[a] = np.max(nushapes[:, a])

    #get the type of the new dataset
    #dtypes = [img.get_data_dtype() for img in imgs]
    if dtype is None:
        dtype = imgs[0].get_data_dtype()

    with h5py.File(file_path, mode) as f:
        try:
            ic = 0
            h5grp = f.create_group(os.path.dirname(h5path))
            h5ds = h5grp.create_dataset(os.path.basename(h5path),
                                        ds_shape, dtype)
            for img in imgs:

                #get the shape of the current image
                nushape = nushapes[ic, :]
                ndims = len(nushape)

                def append_to_dataset(h5ds, idx, data, concat_axis):
                    """
                    @param h5ds: H5py DataSet
                    @param idx: int
                    @param data: ndarray
                    @param concat_axis: int
                    @return:
                    """
                    shape = data.shape
                    ndims = len(shape)

                    if ndims == 1:
                        if concat_axis == 0:
                            h5ds[idx] = data

                    elif ndims == 2:
                        if concat_axis == 0:
                            h5ds[idx ] = data
                        elif concat_axis == 1:
                            h5ds[idx ] = data

                    elif ndims == 3:
                        if concat_axis == 0:
                            h5ds[idx ] = data
                        elif concat_axis == 1:
                            h5ds[idx ] = data
                        elif concat_axis == 2:
                            h5ds[idx ] = data

                #appending the reshaped image into the dataset
                append_to_dataset(h5ds, ic,
                                  np.reshape(img.get_data(), tuple(nushape)),
                                  concat_axis)

                ic += 1

        except ValueError as ve:
            log.error('Error creating group ' + h5path)
            print(str(ve))
