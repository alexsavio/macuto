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
import os.path as op
import sys
import tempfile
import numpy as np
import logging
import subprocess

from ..config import ALLOWED_EXTS
from ..exceptions import FolderNotFound

log = logging.getLogger(__name__)


def get_abspath(folderpath):
    """Returns the absolute path of folderpath.
    If the path does not exist, will raise IOError.
    """
    if not op.exists(folderpath):
        raise FolderNotFound(folderpath)

    return op.abspath(folderpath)


def get_files(folderpath):
    return [op.join(dp, f) for dp, dn, filenames in
            os.walk(folderpath) for f in filenames]


def get_extension(filepath, check_if_exists=False):
    """Return the extension of fpath.

    Parameters
    ----------
    fpath: string
    File name or path

    check_if_exists: bool

    Returns
    -------
    str
    The extension of the file name or path
    """
    if check_if_exists:
        if not op.exists(filepath):
            err = 'File not found: ' + filepath
            log.error(err)
            raise IOError(err)

    try:
        rest, ext = op.splitext(filepath)
        if ext in ALLOWED_EXTS:
            alloweds = ALLOWED_EXTS[ext]
            _, ext2 = op.splitext(filepath)
            if ext2 in alloweds:
                ext = ext2 + ext

        return ext

    except:
        log.error("Unexpected error: ", sys.exc_info()[0])
        raise


def add_extension_if_needed(filepath, ext, check_if_exists=False):
    """Add the extension ext to fpath if it doesn't have it.

    Parameters
    ----------
    filepath: str
    File name or path

    ext: str
    File extension

    check_if_exists: bool

    Returns
    -------
    File name or path with extension added, if needed.
    """
    if not filepath.endswith(ext):
        filepath += ext

    if check_if_exists:
        if not op.exists(filepath):
            err = 'File not found: ' + filepath
            log.error(err)
            raise IOError(err)

    return filepath


def remove_ext(filepath):
    """Removes the extension of the file.

    Parameters
    ----------
    filepath: str
    File path or name

    Returns
    -------
    str
    File path or name without extension
    """
    return filepath[:filepath.rindex(get_extension(filepath))]


def write_lines(filepath, lines):
    """Write the given lines to the file in filepath

    Parameters
    ----------
    filepath: str

    lines: list of str
    """
    try:
        f = open(filepath, 'w')
        f.writelines(lines)
        f.close()
    except IOError as err:
        log.error('Unexpected error: ', err)
        raise
    except:
        log.error('Unexpected error: ', str(sys.exc_info()))
        raise


def grep_one(srch_str, filepath):
    """Return the first line in file defined by filepath
    that contains srch_str

    Parameters
    ----------
    srch_str: str

    filepath: str

    Returns
    ----------
    str
    """
    for line in open(filepath):
        if srch_str in line:
            return line
    return None


def parse_subjects_list(filepath, datadir='', split=':', labelsf=None):
    """Parses a file with a list of: <subject_file>:<subject_class_label>.

    Parameters
    ----------
    filepath: str
    Path to file with a list of: <subject_file>:<subject_class_label>.
    Where ':' can be any split character

    datadir: str
    String to be path prefix of each line of the fname content,
    only in case the lines are relative file paths.

    split: str
    Split character for each line

    labelsf: str
    Path to file with a list of the labels if it is not included in
    fname. It will overwrite the labels from fname.

    Returns
    -------
    [labels, subjs] where labels is a list of labels and subjs a list of
    filepaths
    """
    labels = []
    subjs  = []

    if datadir:
        datadir += op.sep

    try:
        with open(filepath, 'r') as f:
            for s in f:
                line = s.strip().split(split)
                if len(line) == 2:
                    labels.append(np.float(line[1]))
                    subjf = line[0].strip()
                else:
                    subjf = line.strip()

                if not op.isabs(subjf):
                    subjs.append(datadir + subjf)
                else:
                    subjs.append(subjf)

    except:
        log.error("Unexpected error: ", sys.exc_info()[0])
        raise

    if labelsf is not None:
        labels = np.loadtxt(labelsf)

    return [labels, subjs]


def create_subjects_file(filelist, labels, output_file, split=':'):
    """Creates a file where each line is <subject_file>:<subject_class_label>.

    Parameters
    ----------
    filelist: list of str
    List of filepaths

    labels: list of int, str or labels that can be transformed with str()
    List of labels

    output_file: str
    Output file path

    split: str
    Split character for each line

    """
    lines = []
    for s in range(len(filelist)):
        subj = filelist[s]
        lab  = labels[s]
        line = subj + split + str(lab)
        lines.append(line)

    lines = np.array(lines)
    np.savetxt(output_file, lines, fmt='%s')


def join_path_to_filelist(path, filelist):
    """Joins path to each line in filelist

    Parameters
    ----------
    path: str

    filelist: list of str

    Returns
    -------
    list of filepaths
    """
    return [op.join(path, str(item)) for item in filelist]


def remove_all(filelist, folder=''):
    """Deletes all files in filelist

    Parameters
    ----------
    filelist: list of str
    List of the file paths to be removed

    folder: str
    Path to be used as common directory for all file paths in filelist
    """
    if folder:
        try:
            for f in filelist:
                os.remove(f)
        except OSError as err:
            log.error(err)
            pass
    else:
        try:
            for f in filelist:
                os.remove(op.join(folder, f))
        except OSError as err:
            log.error(err)
            pass


def get_folder_subpath(path, folder_depth):
    """
    Returns a folder path of path with depth given by folder_dept:

    Parameters
    ----------
    path: str

    folder_depth: int > 0

    Returns
    -------
    A folder path

    Example
    -------
    >>> get_folder_subpath('/home/user/mydoc/work/notes.txt', 3)
    >>> '/home/user/mydoc'
    """
    if path[0] == op.sep:
        folder_depth += 1

    return '/'.join(path.split('/')[0:folder_depth])


def get_temp_file(dirpath=None, suffix='.nii.gz'):
    """
    Uses tempfile to create a NamedTemporaryFile using
    the default arguments.

    Parameters
    ----------
    dirpath: str
    Directory where it must be created.
    If dir is specified, the file will be created
    in that directory, otherwise, a default directory is used.
    The default directory is chosen from a platform-dependent
    list, but the user of the application can control the
    directory location by setting the TMPDIR, TEMP or TMP
    environment variables.

    suffix: str
    File name suffix.
    It does not put a dot between the file name and the
    suffix; if you need one, put it at the beginning of suffix.

    Returns
    -------
    file object

    Note
    ----
    Please, close it once you have used the file.
    """
    return tempfile.NamedTemporaryFile(dir=dirpath, suffix=suffix)


def get_temp_dir(prefix=None, basepath=None):
    """
    Uses tempfile to create a TemporaryDirectory using
    the default arguments.
    The folder is created using tempfile.mkdtemp() function.

    Parameters
    ----------
    prefix: str
    Name prefix for the temporary folder.

    basepath: str
    Directory where the new folder must be created.
    The default directory is chosen from a platform-dependent
    list, but the user of the application can control the
    directory location by setting the TMPDIR, TEMP or TMP
    environment variables.

    Returns
    -------
    folder object
    """
    if basepath is None:
        return tempfile.TemporaryDirectory(dir=basepath)
    else:
        return tempfile.TemporaryDirectory(prefix=prefix, dir=basepath)


def ux_file_len(filepath):
    """Returns the length of the file using the 'wc' GNU command

    Parameters
    ----------
    filepath: str

    Returns
    -------
    float
    """
    p = subprocess.Popen(['wc', '-l', filepath], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result, err = p.communicate()

    if p.returncode != 0:
        log.error(err)
        raise IOError(err)

    l = result.strip()
    l = int(l.split()[0])
    return l


def count_lines(filepath):
    """Return the number of lines in file in filepath

    Parameters
    ----------
    filepath: str

    Returns
    -------
    int
    """
    statinfo = os.stat(filepath)
    return statinfo.st_size


def file_size(filepath):
    """Returns the length of the file

    Parameters
    ----------
    filepath: str

    Returns
    -------
    float
    """
    return op.getsize(filepath)


def fileobj_size(file_obj):
    """Returns the length of the size of the file

    Parameters
    ----------
    file_obj: file-like object

    Returns
    -------
    float
    """
    file_obj.seek(0, os.SEEK_END)
    return file_obj.tell()
