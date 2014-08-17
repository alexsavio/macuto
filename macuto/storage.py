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
import sys
import shelve
import logging
import h5py
import scipy.io as sio
import pandas as pd

from .files.names import (get_extension,
                          add_extension_if_needed)

log = logging.getLogger(__name__)


def sav_to_pandas_rpy2(input_file):
    """
    SPSS .sav files to Pandas DataFrame through Rpy2

    :param input_file: string

    :return:
    """
    import pandas.rpy.common as com

    w = com.robj.r('foreign::read.spss("%s", to.data.frame=TRUE)' % input_file)
    return com.convert_robj(w)


def sav_to_pandas_savreader(input_file):
    """
    SPSS .sav files to Pandas DataFrame through savreader module

    :param input_file: string

    :return:
    """
    from savReaderWriter import SavReader
    lines = []
    with SavReader(input_file, returnHeader=True) as reader:
        header = next(reader)
        for line in reader:
            lines.append(line)

    return pd.DataFrame(data=lines, columns=header)


def save_variables_to_shelve(file_path, variables):
    """

    Parameters
    ----------
    file_path: str

    variables: dict
        Dictionary with objects. Object name -> object

    Notes
    -----
        Before calling this function, create a varlist this way:

        shelfvars = []
        for v in varnames:
            shelfvars.append(eval(v))

        #to_restore variables from shelf
        my_shelf = shelve.open(filename)
        for key in my_shelf:
           globals()[key]=my_shelf[key]
        my_shelf.close()
    """
    mashelf = shelve.open(file_path, 'n')

    for vn in variables.keys():
        try:
            mashelf[vn] = variables[vn]
        except:
            log.exception('Error shelving variable {0}'.format(vn))
            raise

    mashelf.close()


def save_variables_to_mat(file_path, variables):
    """

    Parameters
    ---------
    file_path: str

    variables: dict
        Dictionary with objects. Object name -> object
    """

    try:
        sio.savemat(file_path, variables, format='4')
    except:
        log.exception('Error saving to {}'.format(file_path))
        raise


def save_variables_to_hdf5(file_path, variables, mode='w', h5path='/'):
    """
    Parameters
    ----------
    file_path: str

    variables: dict
        Dictionary with objects. Object name -> object

    mode: str
        HDF5 file access mode
        See h5py documentation for details.
        Most used here:
        'r+' for read/write
        'w' for destroying then writing
    """
    #h5file = tabs.open_file(outfpath, mode=mode,
    #                        title=os.path.basename(outfpath))
    h5file = h5py.File(file_path, mode)

    h5group = h5file.require_group(h5path)

    try:
        for vn in variables.keys():
            h5group[vn] = variables[vn]

    except:
        log.exception('Error saving to .hdf5: {0}'.format(vn))
        raise

    h5file.close()


class ExportData(object):

    def __init__(self):
        pass

    @staticmethod
    def save_variables(filename, variables):
        """
        Valid extensions '.pyshelf', '.mat', '.hdf5' or '.h5'

        Parameters
        ----------
        filename: str

        variables: dict
            Dictionary varname -> variable
        """
        ext = get_extension(filename).lower()
        out_exts = {'.pyshelf', '.mat', '.hdf5', '.h5'}

        output_file = filename
        if not ext in out_exts:
            output_file = add_extension_if_needed(filename, '.pyshelf')
            ext = get_extension(filename)

        if ext == '.pyshelf':
            save_variables_to_shelve(output_file, variables)

        elif ext == '.mat':
            save_variables_to_mat(output_file, variables)

        elif ext == '.hdf5' or ext == '.h5':
            save_variables_to_hdf5(output_file, variables)

        else:
            log.error('Filename extension {0} not accepted.'.format(ext))

    @staticmethod
    def save_varlist(filename, varnames, varlist):
        """
        Valid extensions '.pyshelf', '.mat', '.hdf5' or '.h5'

        @param filename: string

        @param varnames: list of strings
        Names of the variables

        @param varlist: list of objects
        The objects to be saved
        """
        variables = {}
        for i, vn in enumerate(varnames):
            variables[vn] = eval(varlist[i])

        ExportData.save_variables(filename, variables)


def load_varnames_from_hdf5(file_path, h5path='/'):
    """
    Returns all dataset names from h5path group in file_path.

    Parameters
    ----------
    file_path: string
        HDF5 file path

    h5path: str
        HDF5 group path

    Returns
    -------
    List of variable names contained in file_path
    """
    def walk(group, node_type=h5py.Dataset):
        for node in list(group.values()):
            if isinstance(node, node_type):
                yield node

    h5file  = h5py.File(file_path, mode='r')
    varlist = []
    try:
        h5group = h5file.require_group(h5path)

        for node in walk(h5group):
            varlist.append(node.name)

    except:
        log.exception('ERROR reading .hdf5: {0}'.file_path)
        raise

    h5file.close()

    return varlist


def load_variables_from_hdf5(file_path, h5path='/'):
    """
    Returns all datasets from h5path group in file_path.

    Parameters
    ----------
    file_path: str
        HDF5 file path

    h5path: str
        HDF5 group path

    Returns
    -------
    Dict with variables contained in file_path
    """
    def walk(group, node_type=h5py.Dataset):
        for node in list(group.values()):
            if isinstance(node, node_type):
                yield node

    h5file  = h5py.File(file_path, mode='r')
    vardict = {}
    try:
        h5group = h5file.require_group(h5path)

        for node in walk(h5group):
            node_name = os.path.basename(str(node.name))
            vardict[node_name] = node.value

    except:
        log.exception('ERROR reading .hdf5: {0}'.file_path)
        raise

    h5file.close()

    return vardict




