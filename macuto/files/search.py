# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import re
import os
from glob import glob
from ..strings import search_list, filter_list


def dir_search(regex, wd=None):
    """
    @param regex: string
    @param wd: string
     working directory
    @return:
    """
    if wd is None:
        wd = '.'

    ls = os.listdir(wd)

    filt = re.compile(regex).search
    return filter_list(ls, filt)


def dir_match(regex, wd=None):
    """
    Creates a list of regex matches that result from the match_regex
    of all file names within wd.
    The list of files will have wd as path prefix.

    @param regex: string
    @param wd: string
    working directory
    @return:
    """
    if wd is None:
        wd = ''

    ls = os.listdir(wd)

    filt = re.compile(regex).match
    return filter_list(ls, filt)


def get_file_list(file_dir, regex=None):
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

    if regex is not None:
        file_list = search_list(file_list, regex)

    file_list = [os.path.join(file_dir, fname) for fname in file_list]

    return file_list


def recursive_find(folder_path, regex=None):
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
    if regex is None:
        regex = ''

    return recursive_find_search(folder_path, regex)


def recursive_find_match(folder_path, regex=None):
    """
    Returns absolute paths of files that match the regex within file_dir and
    all its children folders.

    Note: The regex matching is done using the match function
    of the re module.

    Parameters
    ----------
    folder_path: string

    regex: string

    Returns
    -------
    A list of strings.

    """
    if regex is None:
        regex = ''

    outlist = []
    for root, dirs, files in os.walk(folder_path):
        outlist.extend([os.path.join(root, f) for f in files
                        if re.match(regex, f)])

    return outlist


def recursive_find_search(folder_path, regex=None):
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
    if regex is None:
        regex = ''

    outlist = []
    for root, dirs, files in os.walk(folder_path):
        outlist.extend([os.path.join(root, f) for f in files
                        if re.search(regex, f)])

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


def find_match(base_directory, regex=None):
    """
    Uses glob to find all files that match the regex
    in base_directory.

    @param base_directory: string

    @param regex: string

    @return: list

    """
    if regex is None:
        regex = ''

    return glob(os.path.join(base_directory, regex))
