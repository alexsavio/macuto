# coding=utf-8
#------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#------------------------------------------------------------------------------

import os
import sys
import tempfile
import numpy as np
import logging as log
import subprocess

from ..config import ALLOWED_EXTS


def get_extension(fpath, check_if_exists=False):
    """
    @param fpath: string
    File name or path

    @param check_if_exists: bool

    @return: string
    The extension of the file name or path
    """
    if check_if_exists:
        if not os.path.exists(fpath):
            err = 'File not found: ' + fpath
            raise IOError(err)

    try:
        rest, ext = os.path.splitext(fpath)
        if ext in ALLOWED_EXTS:
            alloweds = ALLOWED_EXTS[ext]
            _, ext2 = os.path.splitext(rest)
            if ext2 in alloweds:
                ext = ext2 + ext

        return ext

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
    if not fpath.endswith(ext):
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
    return fname[:fname.rindex(get_extension(fname))]


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
    except IOError as err:
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


def parse_subjects_list(fname, datadir='', split=':', labelsf=None):
    """
    @param fname: string
    Path to file with a list of: <subject_file>:<subject_class_label>.
    Where ':' can be any split character

    @param datadir: string
    String to be path prefix of each line in fname file,
    only in case the lines are relative file paths.

    @param split: string
    Split character for each line

    @param labelsf: string
    Path to file with a list of the labels if it is not included in
    fname. It will overwrite the labels from fname.

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
            if len(line) == 2:
                labels.append(np.float(line[1]))
                subjf = line[0].strip()
            else:
                subjf = line.strip()

            if not os.path.isabs(subjf):
                subjs.append(datadir + subjf)
            else:
                subjs.append(subjf)
        f.close()

    except:
        log.error("Unexpected error: ", sys.exc_info()[0])
        sys.exit(-1)

    if labelsf is not None:
        labels = np.loadtxt(labelsf)

    return [labels, subjs]


def create_subjects_file(filelist, labels, output, split=':'):
    """
    @param filelist:
    @param labels:
    @param output:
    @param split:
    @return:
    """
    lines = []
    for s in range(len(filelist)):
        subj = filelist[s]
        lab  = labels[s]
        line = subj + split + str(lab)
        lines.append(line)

    lines = np.array(lines)
    np.savetxt(output, lines, fmt='%s')


def join_path_to_filelist(path, filelist):
    """
    @param path: string
    @param filelist: list of strings
    @return:
    """
    return [os.path.join(path, str(item)) for item in filelist]


def remove_all(filelist, folder=''):
    """
    @param filelist: list of strings
    @param folder: string
    @return:
    """
    if folder:
        try:
            for f in filelist:
                os.remove(f)
        except OSError as err:
            pass
    else:
        try:
            for f in filelist:
                os.remove(os.path.join(folder, f))
        except OSError as err:
            pass


def get_temp_file(dir=None, suffix='.nii.gz'):
    """
    Uses tempfile to create a NamedTemporaryFile using
    the default arguments.

    @param dir: string
    Directory where it must be created.
    If dir is specified, the file will be created
    in that directory, otherwise, a default directory is used.
    The default directory is chosen from a platform-dependent
    list, but the user of the application can control the
    directory location by setting the TMPDIR, TEMP or TMP
    environment variables.

    @param suffix: string
    File name suffix.
    It does not put a dot between the file name and the
    suffix; if you need one, put it at the beginning of suffix.

    @return: file object

    @note:
    Close it once you have used the file.
    """
    return tempfile.NamedTemporaryFile(dir=dir, suffix=suffix)


def ux_file_len(fname):
    """

    @param fname: string
    @return:
    """
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result, err = p.communicate()

    if p.returncode != 0:
        raise IOError(err)

    l = result.strip()
    l = int(l.split()[0])
    return l


def count_lines(fname):
    """

    @param fname: string
    @return:
    """
    statinfo = os.stat(fname)
    return statinfo.st_size


def file_size(fname):
    """

    @param fname: string
    @return:
    """
    return os.path.getsize(fname)


def fileobj_size(file_obj):
    """

    @param file_obj: file-like object
    @return:
    """
    file_obj.seek(0, os.SEEK_END)
    return file_obj.tell()