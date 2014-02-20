# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import sys
import shelve
import h5py
import scipy.io as sio
import logging as log

from .files import get_extension, add_extension_if_needed


def save_variables_to_shelve(fname, variables):
    """

    @param fname: string
    @param variables: dict
    Dictionary with objects. Object name -> object

    @return:

    @note:
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
    mashelf = shelve.open(fname, 'n')

    for vn in variables.keys():
        try:
            mashelf[vn] = variables[vn]
        except:
            log.error('ERROR shelving: {0}'.format(vn))
            log.error(sys.exc_info())

    mashelf.close()


def save_variables_to_mat(fname, variables):
    """
    @param fname: string
    @param variables: dict
    Dictionary with objects. Object name -> object
    """

    try:
        sio.savemat(fname, variables, format='4')
    except:
        log.error('ERROR saving to' + fname)
        log.error(sys.exc_info())


def save_variables_to_hdf5(fname, variables, mode='w', h5path='/'):
    """
    @param fname: string

    @param variables: dict
    Dictionary with objects. Object name -> object

    @param mode: string
    HDF5 file access mode
    See h5py documentation for details.
    Most used here:
    'r+' for read/write
    'w' for destroying then writing
    """
    #h5file = tabs.open_file(outfpath, mode=mode,
    #                        title=os.path.basename(outfpath))
    h5file = h5py.File(fname, mode)

    h5group = h5file.require_group(h5path)

    try:
        for vn in variables.keys():
            h5group[vn] = variables[vn]

    except:
        log.error('ERROR saving to .hdf5: {0}'.format(vn))
        log.error(sys.exc_info())

    h5file.close()


class ExportData(object):

    def __init__(self):
        pass

    @staticmethod
    def save_variables(filename, variables):
        """
        Valid extensions '.pyshelf', '.mat', '.hdf5' or '.h5'

        @param filename: string

        @param variables: dict
        Dictionary varname -> variable
        """
        ext = get_extension(filename)
        if ext != '.pyshelf' and ext != '.mat' and ext != '.hdf5':
            output_file = add_extension_if_needed(filename, '.pyshelf')
            ext = get_extension(filename)

        if ext == '.pyshelf':
            save_varlist_to_shelve(output_file, variables)

        elif ext == '.mat':
            save_varlist_to_mat(output_file, variables)

        elif ext == '.hdf5' or ext == '.h5':
            save_varlist_to_hdf5(output_file, variables)

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
            variables[vn] = varlist[i]

        ExportData.save_variables(filename, variables)


def load_varnames_from_hdf5(fname, h5path='/'):
    """
    Returns all dataset names from h5path group in fname.

    Parameters
    ----------
    @param fpath: string
    HDF5 file path

    @param h5path: string
    HDF5 group path

    Returns
    -------
    List of variable names contained in fpath
    """
    def walk(group, node_type=h5py.Dataset):
        for node in list(group.values()):
            if isinstance(node, node_type):
                yield node

    h5file  = h5py.File(fname, mode='r')
    varlist = []
    try:
        h5group = h5file.require_group(h5path)

        for node in walk(h5group):
            varlist.append(node.name)

    except:
        log.error('ERROR reading .hdf5: {0}'.fpath)
        log.error(sys.exc_info())

    h5file.close()

    return varlist


def load_variables_from_hdf5(fname, h5path='/'):
    """
    Returns all datasets from h5path group in fname.

    Parameters
    ----------
    @param fpath: string
    HDF5 file path

    @param h5path: string
    HDF5 group path

    Returns
    -------
    Dict with variables contained in fpath
    """
    def walk(group, node_type=h5py.Dataset):
        for node in list(group.values()):
            if isinstance(node, node_type):
                yield node

    h5file  = h5py.File(fname, mode='r')
    vardict = {}
    try:
        h5group = h5file.require_group(h5path)

        for node in walk(h5group):
            node_name = os.path.basename(str(node.name))
            vardict[node_name] = node.value

    except:
        log.error('ERROR reading .hdf5: {0}'.fpath)
        log.error(sys.exc_info())

    h5file.close()

    return vardict




