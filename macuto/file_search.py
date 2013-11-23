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

import re
import os
from .strings import list_search, list_filter


def dir_search (regex, wd='.'):
    """
    @param regex: string
    @param wd: string
     working directory
    @return:
    """
    ls = os.listdir(wd)

    filt = re.compile(regex).search
    return list_filter(ls, filt)


def dir_match (regex, wd='.'):
    """
    Filter
    @param regex: string
    @param wd: string
    working directory
    @return:
    """
    ls = os.listdir(wd)

    filt = re.compile(regex).match
    return list_filter(ls, filt)


def get_file_list(file_dir, search_regex=''):
    """
    Creates a list of files that match the search_regex within file_dir.
    The list of files will have file_dir as path prefix.

    Parameters
    ----------
    @param file_dir:

    @param search_regex:

    Returns:
    --------
    List of paths to files that match the search_regex
    """
    file_list = os.listdir(file_dir)
    file_list.sort()

    if search_regex:
        file_list = list_search(file_list, search_regex)

    file_list = [os.path.join(file_dir, fname) for fname in file_list]

    return file_list


def recursive_find(folder_path, regex=''):
    """
    Returns absolute paths of files that match the regex within file_dir and
    all its children folders.

    Note: The regex matching is done using the search function
    of the re module.

    Parameters
    ----------
    folder_path: string

    regex: string

    Returns
    -------
    A list of strings.

    """
    outlist = []
    for root, dirs, files in os.walk(folder_path):
        outlist.extend([os.path.join(root, f) for f in files if re.search(regex, f)])

    return outlist


def iter_recursive_find(folder_path, *regex):
    '''
    Returns absolute paths of files that match the regexs within file_dir and
    all its children folders.

    This is an iterator function that will use yield to return each set of 
    file_paths in one iteration.

    Will only return value if all the strings in regex match a file name.

    Note: The regex matching is done using the search function
    of the re module.

    Parameters
    ----------
    folder_path: string

    regex: strings

    Returns
    -------
    A list of strings.

    '''
    for root, dirs, files in os.walk(folder_path):
        if len(files) > 0:
            outlist = []
            for f in files:
                for reg in regex:
                    if re.search(reg, f):
                         outlist.append(os.path.join(root, f))
            if len(outlist) == len(regex):
                yield outlist
