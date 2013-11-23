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

import os
import sys
import numpy as np
import logging as log

def get_extension(fpath, check_if_exists=False):
    """
    @param fpath: string
    File name or path

    @param check_if_exists: bool

    @return: string
    The extension of the file name or path
    """
    if check_if_exists:
        if not os.path.exists (fpath):
            err = 'File not found: ' + fpath
            raise IOError(err)

    try:
        s = os.path.splitext(fpath)
        return s[-1]
    except:
        log.error( "Unexpected error: ", sys.exc_info()[0] )
        raise


def add_extension_if_needed(fpath, ext, check_if_exists=False):
    """
    @param fpath: string
    File name or path

    @param ext: string
    File extension

    @param check_if_exists: bool

    @return:
    File name or path with extension added, if needed.
    """
    if fpath.find(ext) < 0:
        fpath += ext

    if check_if_exists:
        if not os.path.exists (fpath):
            err = 'File not found: ' + fpath
            raise IOError(err)

    return fpath


def remove_ext(fname):
    """
    @param fname: string
    File path or name

    @return: string
    File path or name without extension
    """
    if '.nii.gz' in fname:
        return os.path.splitext(os.path.splitext(fname)[0])[0]

    return os.path.splitext(fname)[0]


def write_lines(fname, lines):
    """
    @param fname:
    @param lines:
    @return:
    """
    try:
        f = open(fname, 'w')
        f.writelines(lines)
        f.close()
    except IOError, err:
        log.error ('Unexpected error: ', err)
    except:
        log.error ('Unexpected error: ', str(sys.exc_info()))


def grep_one(srch_str, filepath):
    """
    @param srch_str: string
    @param filepath: string
    @return:
    Returns the first line in file defined by filepath
    that contains srch_str
    """
    for line in open(filepath):
        if srch_str in line:
            return line


def parse_subjects_list(fname, datadir='', split=':'):
    """
    @param fname: string
    File with a list of: <subject_file>:<subject_class_label>.
    Where ':' can be any split character

    @param datadir: string
    String to be path prefix of each line in fname file,
    only in case the lines are relative file paths.

    @param split: string
    Split character for each line

    @return:
    [labels, subjs]
    """
    labels = []
    subjs  = []

    if datadir:
        datadir += os.path.sep

    try:
        f = open(fname, 'r')
        for s in f:
            line = s.strip().split(split)
            labels.append(np.float(line[1]))
            subjf = line[0].strip()
            if not os.path.isabs(subjf):
                subjs.append(datadir + subjf)
            else:
                subjs.append(subjf)
        f.close()

    except:
        log.error("Unexpected error: ", sys.exc_info()[0])
        sys.exit(-1)

    return [labels, subjs]


def create_subjects_file (subjs_list, labels, output, split=':'):
    lines = []
    for s in range(len(subjs_list)):
        subj = subjs_list[s]
        lab  = labels[s]
        line = subj + split + str(lab)
        lines.append(line)

    lines = np.array(lines)
    np.savetxt(output, lines, fmt='%s')