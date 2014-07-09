#!/usr/bin/env python
from __future__ import print_function

import os
import re
import sys
import baker
import dicom
import logging
import subprocess
from collections import defaultdict
from glob import glob
try:
    from path import path
except:
    from pathlib import Path as path

from macuto.files.names get_abspath, get_files

from macuto.dicom.utils import (get_dicom_files, is_dicom_file, call_dcm2nii,
                                anonymize_dicom_file, group_dicom_files,
                                anonymize_dicom_file_dcmtk)

from macuto.exceptions import LoggedError, FolderAlreadyExists
from macuto.config import (DICOM_FILE_EXTENSIONS,
                           OUTPUT_DICOM_EXTENSION)


#logging config
log = logging.getLogger(__name__)

#santiago search idregex
#santiago_idregex = '[N|P]?\\d\\d*-?\\d?$'


class EmptySubjectFolder(LoggedError):
    pass


@baker.command(default=True,
               shortopts={'i': 'subjfolder',
                          'r': 'idregex',
                          'f': 'not_rename_folder'})
def subject(subjfolder, idregex='', not_rename_folder=False):
    """Anonymizes the entire subject folder.
    First renames the main folder to the acquisition ID, then 
    it looks inside for subdirectories with DICOM files and
    anonymizes each of them.

    :param subjfolder: Path to the subject folder

    :param idregex: Regex to search for ID in folder name

    :param not_rename_folder: If this flag is set, will not rename
                              subjects' folder
    """
    log.info('anonymizer.py subject {0} {1} {2}'.format(subjfolder,
                                                        idregex,
                                                        not_rename_folder))

    if sys.version_info[0] == 2:
        subjfolder = subjfolder.decode('utf-8')

    if not_rename_folder:
        subj_folder = subjfolder
    else:
        try:
            subj_folder = folder_name(subjfolder, idregex=idregex)
        except:
            raise

        if not subj_folder:
            raise EmptySubjectFolder(log, 'Got empty subj_folder from folder_name '
                                          'renaming function.')

    log.info('Anonymizing folder: ' + subjfolder + ' to ' + subj_folder)

    acqfolders = path(subj_folder).listdir()
    for foldr in acqfolders:
        if foldr.isdir():
            try:
                dicom_folder(foldr)
            except Exception as e:
                raise LoggedError(log, str(e))


@baker.command(shortopts={'input_folder': 'i',
                          'output_folder': 'o',
                          'header_field': 'h',
                          'overwrite': 'w'})
def batch(input_folder, output_folder, header_field='PatientID',
          overwrite=False):
    """Will get all DICOMs inside the input_folder and copy them
    separated and organized by the different header_field values
    found in all these DICOM files.
    After that, will convert the files to nifti using MRICron dcm2nii

    :param input_folder: str

    :param output_folder: str

    :param header_field: str

    :param overwrite: bool
    If True and the output_folder exists, will remove its files.
    """
    log.info('{0} {1} {2}'.format(input_folder, output_folder, header_field))

    if os.path.exists(output_folder):
        if not overwrite:
            if os.listdir(output_folder):
                msg = 'Please change it or empty it.'
                raise FolderAlreadyExists(log, output_folder, msg)
        else:
            import shutil
            shutil.rmtree(output_folder)

    log.info('Listing DICOM all files in {0}.'.format(input_folder))
    dicoms = get_dicom_files(input_folder)

    log.info('Grouping DICOM files by subject.')
    dicom_sets = group_dicom_files(dicoms, header_field)

    try:
        new_dicom_sets = create_dicom_subject_folders(output_folder, dicom_sets)
    except Exception as exc:
        raise LoggedError(log, 'ERROR create_dicom_subject_folders: '
                               '{0}'.format(str(exc)))

    for dcm_set in new_dicom_sets:
        try:
            dicom_to_nii(os.path.join(output_folder, dcm_set))
        except Exception as exc:
            raise LoggedError(log, 'ERROR dicom_to_nii {0}. {1}'.format(dcm_set,
                                                                        str(exc)))


@baker.command(shortopts={'acqfolder': 'i'})
def dicom_folder(acqfolder):
    """Anonymizes all DICOM files one step within acqfolder.
    First it removes the personal header content of the DICOM files,
    then if converts the DICOMs to NifTI and finally it renames the
    DICOM file names.

    :param acqfolder: Path to the subject's acquisition folder
    """
    log.info('{0}'.format(acqfolder))

    acqfolder = get_abspath(acqfolder)

    try:
        dicom_headers(acqfolder)
        dicom_to_nii(acqfolder)
        file_names(acqfolder)
    except Exception as e:
        raise LoggedError(log, 'Cannot anonymize folder or '
                               'file {0}.'.format(acqfolder))


@baker.command(shortopts={'subjfolder': 'i', 'newfolder': 'o',
                          'idregex': 'r'})
def folder_name(subjfolder, newfolder=None, idregex=None):
    """Moves the subjfolder either to newfolder, or
    to a folder named as the PatientID in the DICOM
    files within it, in the same base path.

    :param subjfolder: Path to the subject folder

    :param newfolder: Path to the new folder

    :param idregex: Regex to search for ID in folder name
    """
    log.info('{0} {1} {2}'.format(subjfolder.encode('utf-8'), newfolder,
                                  idregex))

    subjfolder = get_abspath(subjfolder)

    if newfolder is not None:
        log.info('Moving: ' + subjfolder + ' -> ' + newfolder)
        path(subjfolder).move(newfolder)

        return newfolder

    #newfolder is None, let's find other name
    #get name of the base folder
    basedir = subjfolder.dirname()
    if subjfolder[-1] == os.sep:
        basedir = basedir.dirname()

    if idregex is not None:
        subjid = re.search(idregex, subjfolder.basename()).group(0)
        if len(subjid) > 3:
            newfolder = subjid
        else:
            raise LoggedError(log, 'Could not find "{0}" on folder name '
                                   '{1}.'.format(idregex, basedir))

    #try to guess new folder name from DICOM headers
    if newfolder is None:
        log.info('Reading internal DICOMs PatientID to get new folder name.')
        newfolder = get_patient_mri_id(subjfolder)

    if newfolder is None:
        log.error('Could not find a folder name for {0} from DICOMs.'.format(subjfolder))
        return subjfolder

    #else:
    #move the subjfolder to the new folder if it has a different name
    newfolder = os.path.join(basedir, newfolder)
    if subjfolder != newfolder:
        log.info('Moving: ' + subjfolder + ' -> ' + newfolder)
        path(subjfolder).move(newfolder)

    return newfolder


@baker.command(shortopts={'acqfolder': 'i'})
def file_names(acqfolder):
    """Renames the .IMA and .DICOM files within a subjfolder's acquisition
    to 0001.IMA, 0002.IMA...

    :param acqfolder: Path to the folder where a set of
                     .IMA, .DICOM or .DCM files are stored.
    """
    log.info('anonymizer.py file_names {0}'.format(acqfolder))

    subj_path = path(acqfolder)

    done = -1
    for ext in DICOM_FILE_EXTENSIONS:
        file_lst = subj_path.glob('*' + ext)
        if file_lst:
            rename_file_group_to_serial_nums(file_lst)
            done = 0

    return done


@baker.command(shortopts={'acqpath': 'i'})
def dicom_headers(acqpath):
    """Anonymizes all the DICOM files within acqpath in case it is a folder,
    else if it is a file path, will anonymize the file.

    Anonymizing means: putting nonsense information into tags:
    PatientName, PatientAddress and PatientBirthDate.

    :param acqfolder: Path to the folder where a set of .IMA, .DICOM or .DCM
                      files are stored. Can also be the path to only one DICOM file.
    """
    log.info('anonymizer.py dicom_headers {0}'.format(acqpath))

    subj_path = path(acqpath)
    log.info('Anonymizing DICOM files in folder {0}'.format(subj_path))

    if subj_path.isfile():
        anonymize_dicom_file(subj_path)
    else:
        for ext in DICOM_FILE_EXTENSIONS:
            file_lst = subj_path.glob('*' + ext)
            for dcm_file in file_lst:
                try:
                    anonymize_dicom_file(dcm_file)
                except Exception as e:
                    raise LoggedError(log, 'Could not anonymize file ' + dcm_file)


@baker.command(params={"acqpath": "Path to the subject's acquisition folder with DICOM files"},
               shortopts={'acqpath': 'i',
                          'use_known_extensions': 'e'})
def dicom_to_nii(acqpath, use_known_extensions=False):
    """Uses dcm2nii to convert all DICOM files within acqpath to NifTI.
    """
    log.info('anonymizer.py dicom_to_nii {0}'.format(acqpath))

    subj_path = get_abspath(acqpath)

    if subj_path.isfile():
        call_dcm2nii(subj_path)

    else:
        if use_known_extensions:
            for ext in DICOM_FILE_EXTENSIONS:
                regex = '*' + ext
                if subj_path.glob(regex):
                    call_dcm2nii(subj_path.joinpath(regex))
        else:
            regex = '*'
            call_dcm2nii(subj_path.joinpath(regex))


def get_all_patient_mri_ids(subjfolder):
    """Recursively looks for DICOM files in subjfolder, will
    return the value of the first PatientID tag it finds.
    """
    assert(os.path.exists(subjfolder))

    subj_ids = set()

    for ext in DICOM_FILE_EXTENSIONS:
        file_lst = []
        file_lst.extend(glob(os.path.join(subjfolder, '*', '*' + ext)))
        file_lst.extend(glob(os.path.join(subjfolder, '*' + ext)))

        if file_lst:
            for dcm_file in file_lst:
                plan = dicom.read_file(dcm_file)
                if hasattr(plan, 'PatientID'):
                    if plan.PatientID is not None:
                        subj_ids.add(plan.PatientID)
    return subj_ids


def get_patient_mri_id(subjfolder):
    """Recursively looks for DICOM files in subjfolder, will
    return the value of the first PatientID tag it finds.
    """
    assert(os.path.exists(subjfolder))

    for ext in DICOM_FILE_EXTENSIONS:
        file_lst = []
        file_lst.extend(glob(os.path.join(subjfolder, '*', '*' + ext)))
        file_lst.extend(glob(os.path.join(subjfolder, '*' + ext)))

        if file_lst:
            dcm_file = file_lst[0]
            plan = dicom.read_file(dcm_file)
            if hasattr(plan, 'PatientID'):
                if plan.PatientID is not None:
                    return plan.PatientID
            else:
                continue
    return None


def rename_file_group_to_serial_nums(file_lst):
    """Will rename all files in file_lst to a padded serial
    number plus its extension

    :param file_lst: list of path.py paths
    """
    file_lst.sort()
    c = 1
    for f in file_lst:
        dirname = path.abspath(f.dirname())
        fdest = f.joinpath(dirname, "{0:04d}".format(c) + OUTPUT_DICOM_EXTENSION)
        log.info('Renaming {0} to {1}'.format(f, fdest))
        f.rename(fdest)
        c += 1


if __name__ == '__main__':
    baker.run()
    #batch('/home/alexandre/Desktop/new_ariadna', '/home/alexandre/Desktop/new_ariadna_nii', overwrite=True)
