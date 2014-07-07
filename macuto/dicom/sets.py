
import os
import logging
from collections import defaultdict

from ..config import (DICOM_FILE_EXTENSIONS,
                      OUTPUT_DICOM_EXTENSION)
from ..exceptions import LoggedError, ValueError
from ..files.names import get_abspath
from ..more_collections import ItemSet
from .utils import get_dicomfiles, get_dicom_file_paths

log = logging.getLogger(__name__)


class DicomFileList(ItemSet):

    def __init__(self, folders, store_metadata=False, header_fields=None):
        """

        :param folders: str or list of str
        Paths to folders containing DICOM files.
        If None, won't look for files anywhere.

        :param store_metadata: bool

        :param header_fields: set of strings
        Set of header fields to be stored for each DICOM file.
        If store_metadata is False, this won't be used.

        """

        self.items = []
        self.store_metadata = store_metadata
        self.header_fields = header_fields

        if folders is not None:
            if isinstance(folders, list):
                self.from_list(folders)
            elif isinstance(folders, str):
                self.add_folder(folders)
            else:
                raise ValueError('ValueError: Could not recognize folders '
                                 'argument value.')

    def add_folder(self, folder):
        """

        :param folder: str
         Path to a new folder containing Dicom files.
        :return:
        """
        if self.store_metadata:
            try:
                new_filelst = get_dicomfiles(folder, self.header_fields)
            except LoggedError as lerr:
                raise lerr
        else:
            new_filelst = get_dicom_file_paths(folder)

        if self.items:
            self.items.extend(new_filelst)
        else:
            self.items = new_filelst

    def from_list(self, folders):
        """

        :param folders: list of str

        :return
        """
        self.items = []
        for folder in folders:
            self.add_folder(folder)

    def from_set(self, fileset):
        self.items = list(fileset)

    def to_list(self):
        return list(self.items)

    def to_folder(self, output_path):
        """

        :param output_path:
        :return:
        """
        raise NotImplementedError
        #TODO


# def batch(input_folder, output_folder, header_field='PatientID',
#           overwrite=False):
#     """Will get all DICOMs inside the input_folder and copy them
#     separated and organized by the different header_field values
#     found in all these DICOM files.
#     After that, will convert the files to nifti using MRICron dcm2nii
#
#     :param input_folder: str
#
#     :param output_folder: str
#
#     :param header_field: str
#
#     :param overwrite: bool
#     If True and the output_folder exists, will remove its files.
#     """
#     log.info('{0} {1} {2}'.format(input_folder, output_folder, header_field))
#
#     if os.path.exists(output_folder):
#         if not overwrite:
#             if os.listdir(output_folder):
#                 msg = 'Please change it or empty it.'
#                 raise FolderAlreadyExists(output_folder, msg)
#         else:
#             import shutil
#             shutil.rmtree(output_folder)
#
#     log.info('Listing DICOM all files in {0}.'.format(input_folder))
#     dicoms = get_dicom_files(input_folder)
#
#     log.info('Grouping DICOM files by subject.')
#     dicom_sets = group_dicom_files(dicoms, header_field)
#
#     try:
#         new_dicom_sets = create_dicom_subject_folders(output_folder, dicom_sets)
#     except Exception as exc:
#         raise LoggedError('ERROR create_dicom_subject_folders: '
#                           '{0}'.format(str(exc)))
#
#     for dcm_set in new_dicom_sets:
#         try:
#             dicom_to_nii(os.path.join(output_folder, dcm_set))
#         except Exception as exc:
#             raise LoggedError('ERROR dicom_to_nii {0}. {1}'.format(dcm_set,
#                                                                    str(exc)))


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


def rename_file_group_to_serial_nums(file_lst):
    """Will rename all files in file_lst to a padded serial
    number plus its extension

    :param file_lst: list of path.py paths
    """
    file_lst.sort()
    c = 1
    for f in file_lst:
        dirname = get_abspath(f.dirname())
        fdest = f.joinpath(dirname, "{0:04d}".format(c) + OUTPUT_DICOM_EXTENSION)
        log.info('Renaming {0} to {1}'.format(f, fdest))
        f.rename(fdest)
        c += 1

if __name__ == '__main__':
    from macuto.dicom.sets import DicomFileList

    datadir_hd = '/media/alexandre/cobre/santiago/test' #HD 4.2GB in 9981 DICOMS
    #%timeit dicoms = DicomFileList(datadir_hd, store_metadata=True)
    #1 loops, best of 3: 38.4 s per loop

    datadir_ssd = '/scratch/santiago_test' #SSD 4.2GB in 9981 DICOMS
    #%timeit dicoms = DicomFileList(datadir_ssd, store_metadata=True)
    #1 loops, best of 3: 38 s per loop

    datadir = '/scratch/santiago'
    from macuto.dicom.sets import DicomFileList
    dicoms = DicomFileList(datadir, store_metadata=True)