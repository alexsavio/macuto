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

from macuto.config import (DICOM_FILE_EXTENSIONS,
                           OUTPUT_DICOM_EXTENSION)

from macuto.exceptions import LoggedError


#logging config
log = logging.getLogger(__name__)

#santiago search idregex
#santiago_idregex = '[N|P]?\\d\\d*-?\\d?$'


class EmptySubjectFolder(LoggedError):
    pass


class FolderDoesNotExist(LoggedError):
    pass


class OutputFolderAlreadyExists(LoggedError):
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
            raise EmptySubjectFolder('Got empty subj_folder from folder_name '
                                     'renaming function.')

    log.info('Anonymizing folder: ' + subjfolder + ' to ' + subj_folder)

    acqfolders = path(subj_folder).listdir()
    for foldr in acqfolders:
        if foldr.isdir():
            try:
                dicom_folder(foldr)
            except Exception as e:
                raise LoggedError(str(e))


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
                msg = 'Output folder {0} is not empty. ' \
                      'Please change it or empty it.'.format(output_folder)
                raise OutputFolderAlreadyExists(msg)
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
        raise LoggedError('ERROR create_dicom_subject_folders: '
                          '{0}'.format(str(exc)))

    for dcm_set in new_dicom_sets:
        try:
            dicom_to_nii(os.path.join(output_folder, dcm_set))
        except Exception as exc:
            raise LoggedError('ERROR dicom_to_nii {0}. {1}'.format(dcm_set,
                                                                   str(exc)))


def create_dicom_subject_folders(out_path, dicom_sets):
    """

    :param out_path: str
     Path to the output directory

    :param dicom_sets: dict of {str: list of strs}
     Groups of dicom files
    """
    import shutil

    try:
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        new_groups = defaultdict(list)
        for group in dicom_sets:
            group_path = os.path.join(out_path, str(group))
            os.mkdir(group_path)

            group_dicoms = dicom_sets[group]
            for idx, dcm in enumerate(group_dicoms):
                num = str(idx).zfill(5)
                new_dcm = os.path.join(group_path, num + DICOM_FILE_EXTENSIONS[0].lower())
                log.info('Copying {0} -> {1}'.format(dcm, new_dcm))
                shutil.copyfile(dcm, new_dcm)
                new_groups[group].append(new_dcm)

        return new_groups

    except:
        raise


def group_dicom_files(dicom_paths, hdr_field='PatientID'):
    """

    :param dicom_paths: str
    :return: dict of dicom_paths
    """
    dicom_groups = defaultdict(list)
    for dcm in dicom_paths:
        hdr = dicom.read_file(dcm)

        try:
            group_key = getattr(hdr, hdr_field)
        except:
            raise

        dicom_groups[group_key].append(dcm)

    return dicom_groups


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
        raise LoggedError('Cannot anonymize folder or '
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
            raise LoggedError('Could not find "{0}" on folder name '
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
                    raise LoggedError('Could not anonymize file ' + dcm_file)


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


def call_dcm2nii(input_path):
    """

    :param input_path: str
    :return:
    """
    try:
        log.info('dcm2nii {0}'.format(input_path))
        return subprocess.call('dcm2nii {0}'.format(input_path),
                               shell=True)

    except Exception as e:
        raise LoggedError('Error calling dcm2nii on {0}. {1}'.format(input_path,
                                                                     str(e)))


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


def anonymize_dicom_file(dcm_file, remove_private_tags=False,
                         remove_curves=False):
    """Anonymizes the given dcm_file.

    Anonymizing means: putting nonsense information into tags:
    PatientName, PatientAddress and PatientBirthDate.

    :param acqfolder: path.py path
    Path to the DICOM file.
    """
    assert(dcm_file.isfile())

    # Load the current dicom file to 'anonymize'
    plan = dicom.read_file(dcm_file)

    plan.PatientName = 'Anonymous'
    plan.PatientAddress = 'North Pole'

    # Define call-back functions for the dataset.walk() function
    def PN_callback(ds, data_element):
        """Called from the dataset "walk" recursive function for all data elements."""
        if data_element.VR == "PN":
            data_element.value = 'Anonymous'
    def curves_callback(ds, data_element):
        """Called from the dataset "walk" recursive function for all data elements."""
        if data_element.tag.group & 0xFF00 == 0x5000:
            del ds[data_element.tag]
    
    # Remove patient name and any other person names
    plan.walk(PN_callback)

    # Remove data elements (should only do so if DICOM type 3 optional) 
    # Use general loop so easy to add more later
    # Could also have done: del ds.OtherPatientIDs, etc.
    #for name in ['OtherPatientIDs']:
    #    if name in plan:
    #        delattr(ds, name)

    # Same as above but for blanking data elements that are type 2.
    for name in ['PatientsBirthDate']:
        if name in plan:
            plan.data_element(name).value = ''

    # Remove private tags if function argument says to do so. Same for curves
    if remove_private_tags:
        plan.remove_private_tags()
    if remove_curves:
        plan.walk(curves_callback)

    # write the 'anonymized' DICOM out under the new filename
    plan.save_as(dcm_file)


def anonymize_dicom_file_dcmtk(dcm_file):
    """Anonymizes the given dcm_file.

    Anonymizing means: putting nonsense information into tags:
    PatientName, PatientAddress and PatientBirthDate.

    :param acqfolder: path.py path
    Path to the DICOM file.
    """
    assert(dcm_file.isfile())

    subprocess.call('dcmodify --modify PatientName=Anonymous ' + dcm_file, 
                    shell=True)
    subprocess.call('dcmodify --modify PatientBirthDate=17000101 ' + dcm_file, 
                    shell=True)
    subprocess.call('dcmodify --modify PatientAddress=North Pole ' + dcm_file, 
                    shell=True)

    path(dcm_file + '.bak').remove()


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


def get_abspath(folderpath):
    """Returns the absolute path of folderpath.
    If the path does not exist, will raise IOError.
    """
    #if not os.path.exists(folderpath):
    #    raise FolderDoesNotExist('Acquisition folder {0} not '
    #                             'found.'.format(folderpath))

    try:
        return path(folderpath).abspath()
    except:
        raise


def get_files(dirpath):
    return [os.path.join(dp, f) for dp, dn, filenames in
            os.walk(dirpath) for f in filenames]


def get_dicom_files(dirpath):
    return [os.path.join(dp, f) for dp, dn, filenames in
            os.walk(dirpath) for f in filenames
            if is_dicom_file(os.path.join(dp, f))]


def is_dicom_file(filepath):
    """

    :param filepath: string
     Path to DICOM file

    :return: bool
    """
    filename = path(filepath).basename()
    if filename == 'DICOMDIR':
        return False

    try:
        _ = dicom.read_file(filepath)
    except:
        return False

    return True


if __name__ == '__main__':
    #baker.run()
    batch('/home/alexandre/Desktop/new_ariadna', '/home/alexandre/Desktop/new_ariadna_nii', overwrite=True)