
import os
import logging
import numpy as np

from collections import defaultdict
from ..more_collections import DefaultOrderedDict
from ..files.names import get_abspath
from ..config import (DICOM_FILE_EXTENSIONS,
                      OUTPUT_DICOM_EXTENSION)
from .comparison import DicomFileDistance
from ..exceptions import LoggedError
from .utils import DicomFile, group_dicom_files, get_dicom_files

log = logging.getLogger(__name__)


class DicomFilesClustering(object):
    """A self-organizing set file of DICOM files.
    It uses macuto.dicom.comparison.DicomDistanceMeasure to compare
    all DICOM files within a set of folders and create clusters of DICOM files
    that are similar.

    This has been created to automatically generate sets of files for different
    subjects.

    Set the fields/weights in DICOM_FIELD_WEIGHTS to adjust this.
    """

    def __init__(self, folders):
        """

        :param folders:
        :return:
        """
        self._folders = []
        self._files = []
        self._file_dists = None
        self._subjs = DefaultOrderedDict(list)

        self.add_folder(folders)

    def add_folder(self, folder_path):

        if isinstance(folder_path, list):
            for folder in folder_path:
                self._folders.extend(get_abspath(folder))
        else:
            self._folders.append(get_abspath(folder_path))

        self._update()

    def _update(self):
        self._update_file_list()
        self._calculate_file_distances()
        #self._update_subjs_dict()
        #self._reorder_file_list()

    def _update_file_list(self):

        self._files = []
        for folder in self._folders:
            log.info('Detecting DICOM files within {0}.'.format(folder))

            self._files.extend(get_dicom_files(folder))

    def _calculate_file_distances(self):

        log.info('Calculating distance between DICOM files.')
        n_files = len(self._files)

        dist_method = DicomFileDistance()

        self._file_dists = np.zeros((n_files, n_files))

        for idxi in range(n_files):
            dist_method.dcmf1 = DicomFile(self._files[idxi])

            for idxj in range(idxi+1, n_files):
                dist_method.dcmf2 = DicomFile(self._files[idxj])

                if idxi != idxj:
                    self._file_dists[idxi, idxj] = dist_method.transform()


    def _update_subjs_dict(self):
        raise NotImplementedError
        #TODO

    def _reorder_file_list(self):
        raise NotImplementedError
        #TODO
        #mylist=['a','b','c','d','e']
        #myorder=[3,2,0,1,4]
        #mylist = [ mylist[i] for i in myorder]
        #print mylist


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

    from macuto.dicom.sets import DicomFilesClustering

    wd = '/media/alexandre/cobre/santiago/data'

    dcmclusters = DicomFilesClustering(wd)