
import os
import logging
from collections import defaultdict, namedtuple

import macuto.dicom.utils as du

from ..config import (DICOM_FILE_EXTENSIONS,
                      OUTPUT_DICOM_EXTENSION)
from ..exceptions import FolderNotFound
from ..files.names import get_abspath
from ..more_collections import ItemSet

log = logging.getLogger(__name__)


class DicomFileSet(ItemSet):
    """Class to store unique absolute dicom file paths"""

    def __init__(self, folders=None):
        """
        :param folders: str or list of strs
        Path or paths to folders to be searched for Dicom files
        """
        self.items = []

        if folders is not None:
            try:
                self._store_dicom_paths(folders)
            except FolderNotFound as fe:
                log.error('Error storing dicom file paths. {}'.format(fe.msg))

    def _store_dicom_paths(self, folders):
        """Search for dicoms in folders and save file paths into
        self.dicom_paths set.

        :param folders: str or list of str
        """
        if isinstance(folders, str):
            folders = [folders]

        for folder in folders:

            if not os.path.exists(folder):
                raise FolderNotFound(folder)

            self.items.extend(list(du.find_all_dicom_files(folder)))

    def from_folders(self, folders):
        """
        Restart the self.items and stores all dicom file paths found
        within folders

        Parameters
        ----------
        folders: str or list of str
        """
        self.items = []
        self._store_dicom_paths(folders)

    def from_set(self, fileset, check_if_dicoms=True):
        """Overwrites self.items with the given set of files.
        Will filter the fileset and keep only Dicom files.

        Parameters
        ----------
        fileset: iterable of str
        Paths to files

        check_if_dicoms: bool
        Whether to check if the items in fileset are dicom file paths
        """
        if check_if_dicoms:
            self.items = []
            for f in fileset:
                if du.is_dicom_file(f):
                    self.items.append(f)
        else:
            self.items = fileset

    def update(self, dicomset):
        """Update this set with the union of itself and dicomset.

        Parameters
        ----------
        dicomset: DicomFileSet
        """
        if not isinstance(dicomset, DicomFileSet):
            raise ValueError('Given dicomset is not a DicomFileSet.')

        self.items = list(set(self.items).update(dicomset))

    def copy_files_to_other_folder(self, output_folder, rename_files=True,
                                   mkdir=True, verbose=False):
        """
        Copies all files within this set to the output_folder

        Parameters
        ----------
        output_folder: str
        Path of the destination folder of the files

        rename_files: bool
        Whether or not rename the files to a sequential format

        mkdir: bool
        Whether to make the folder if it does not exist

        verbose: bool
        Whether to print to stdout the files that are beind copied
        """
        import shutil

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if not rename_files:
            for dcmf in self.items:
                outf = os.path.join(output_folder, os.path.basename(dcmf))
                if verbose:
                    print('{} -> {}'.format(dcmf, outf))
                shutil.copyfile(dcmf, outf)
        else:
            n_pad = len(self.items)+2
            for idx, dcmf in enumerate(self.items):
                outf = '{number:0{width}d}.dcm'.format(width=n_pad, number=idx)
                outf = os.path.join(output_folder, outf)
                if verbose:
                    print('{} -> {}'.format(dcmf, outf))
                shutil.copyfile(dcmf, outf)


class DicomGenericSet(DicomFileSet):

    def __init__(self, folders, read_metadata=True, header_fields=None):
        """

        :param folders: str or list of strs
        Path or paths to folders to be searched for Dicom files

        :param read_metadata: bool
        If True, will either make a list of DicomFiles, or
        a simple DICOM header (namedtuples) with the fields specified
        in header_fields.

        :param header_fields: set of strings
        Set of header fields to be read from each DICOM file in a DicomHeader.
        If store_metadata is False, this won't be used. Else and if this is
        None, will store the whole DicomFile.
        """
        super(DicomGenericSet, self).__init__(self, folders)
        self.read_dcm = self.get_dcm_reader(read_metadata, header_fields)

    @staticmethod
    def get_dcm_reader(store_metadata=True, header_fields=None):
        """
        Creates a lambda function to read DICOM files.
        If store_store_metadata is False, will only return the file path.
        Else if you give header_fields, will return only the set of of
        header_fields within a DicomFile object or the whole DICOM file if
        None.

        :return: function
        This function has only one parameter: file_path
        """
        if not store_metadata:
            build_dcm = lambda fpath: fpath
        else:
            if header_fields is None:
                build_dcm = lambda fpath: du.DicomFile(fpath)
            else:
                dicom_header = namedtuple('DicomHeader', header_fields)
                build_dcm = lambda fpath: dicom_header.\
                    _make(du.DicomFile(fpath).get_attributes(header_fields))

        return build_dcm

    def scrape_all_files(self):
        """
        Generator that yields one by one the return value for self.read_dcm
        for each file within this set
        """
        try:
            for dcmf in self.items:
                yield self.read_dcm(dcmf)
        except Exception as exc:
            log.exception('Error reading DICOM file: {} '
                          '\n {}'.format(dcmf, str(exc)))

    # def scrape_dicom_pairs(self):
    #     """
    #     Generator that yields a 2-tuple with the return values of self.read_dcm
    #     of all possible pairs of files within this set.
    #     This is used for comparison between the files.
    #     """
    #     n_files = len(self.items)
    #     try:
    #         for idx1 in range(n_files):
    #             yield self.read_dcm(dcmf)
    #     except Exception as exc:
    #         log.exception('Error reading DICOM file: {} '
    #                       '\n {}'.format(dcmf, str(exc)))

    def __iter__(self):
        return self

    def __next__(self):
        return self.scrape_all_files()

    def next(self):
        return self.__next__()

    def __getitem__(self, item):
        if hasattr(self.items, '__getitem__'):
            return self.read_dcm(self.items[item])
        else:
            raise log.exception('Item set has no __getitem__ implemented.')


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
#         new_dicom_sets = create_dicom_subject_folders(output_folder,
#                                                       dicom_sets)
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
        log.exception('Creating DICOM subject folders.')


def rename_file_group_to_serial_nums(file_lst):
    """Will rename all files in file_lst to a padded serial
    number plus its extension

    :param file_lst: list of path.py paths
    """
    file_lst.sort()
    c = 1
    for f in file_lst:
        dirname = get_abspath(f.dirname())
        fdest = f.joinpath(dirname, "{0:04d}".format(c) +
                           OUTPUT_DICOM_EXTENSION)
        log.info('Renaming {0} to {1}'.format(f, fdest))
        f.rename(fdest)
        c += 1

if __name__ == '__main__':
    from macuto.config import DICOM_FIELD_WEIGHTS
    from macuto.dicom.sets import DicomsGenericSet

    #HD 4.2GB in 9981 DICOMS
    datadir_hd = '/media/alexandre/cobre/santiago/test'
    #%timeit dicoms = DicomFileList(datadir_hd, store_metadata=True)
    #1 loops, best of 3: 38.4 s per loop

    #SSD 4.2GB in 9981 DICOMS
    datadir_ssd = '/scratch/santiago_test'
    #%timeit dicoms = DicomFileList(datadir_ssd, store_metadata=True)
    #1 loops, best of 3: 38 s per loop

    datadir = '/scratch/santiago'
    header_fields = tuple(DICOM_FIELD_WEIGHTS.keys())

    dicoms = DicomsGenericSet(datadir, store_metadata=True,
                              header_fields=header_fields)

