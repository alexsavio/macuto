__author__ = 'alexandre'

import os
import logging
import numpy as np

from ..more_collections import DefaultOrderedDict, merge_dict_of_lists
from ..files.names import get_folder_subpath
from ..config import DICOM_FIELD_WEIGHTS
from ..classification.distance import DistanceMeasure

from .utils import DicomFile
from .sets import DicomFileSet

log = logging.getLogger(__name__)


class DicomFileDistance(DistanceMeasure):

    import Levenshtein

    similarity_measure = Levenshtein.ratio

    def __init__(self, field_weights=DICOM_FIELD_WEIGHTS):
        self.dcmf1 = None
        self.dcmf2 = None
        self.field_weights = field_weights
        self.inv_sum_weights = 1/sum(tuple(field_weights.values()))

    def fit(self, dcm_file1, dcm_file2):
        """

        :param dcm_file1: str (path to file) or DicomFile or namedtuple

        :param dcm_file2: str (path to file) or DicomFile or namedtuple
        """
        self.set_dicom_file1(dcm_file1)
        self.set_dicom_file2(dcm_file2)

    def set_dicom_file1(self, dcm_file):
        """
        :param dcm_file: str (path to file) or DicomFile or namedtuple
        """
        self.dcmf1 = self._read_dcmfile(dcm_file)

    def set_dicom_file2(self, dcm_file):
        """
        :param dcm_file: str (path to file) or DicomFile or namedtuple
        """
        self.dcmf2 = self._read_dcmfile(dcm_file)

    @staticmethod
    def _read_dcmfile(dcm_file):
        """

        :param dcm_file:
        :return:
        """
        if isinstance(dcm_file, str):
            return DicomFile(dcm_file)
        else:
            return dcm_file

    def fit_transform(self, file_path1, file_path2):
        self.fit(file_path1, file_path2)
        return self.transform()

    def transform(self):

        if self.dcmf1 is None or self.dcmf2 is None:
            return np.inf

        if len(self.field_weights) == 0:
            log.exception('Field weights are not set.')

        field_weights = self.field_weights.copy()

        dist = 0
        for field_name in list(field_weights.keys()):

            str1, str2 = '', ''
            try:
                str1 = str(getattr(self.dcmf1, field_name))
            except AttributeError as ae:
                log.exception('Error reading attribute {} from '
                              'file {}'.format(field_name,
                                               self.dcmf1.file_path))
                field_weights.pop(field_name, '')

            try:
                str2 = str(getattr(self.dcmf2, field_name))
            except AttributeError as ae:
                log.exception('Error reading attribute {} from '
                              'file {}'.format(field_name,
                                               self.dcmf2.file_path))
                field_weights.pop(field_name, '')

            if not str1 or not str2:
                continue

            sum_weights = sum(tuple(field_weights.values()))
            if sum_weights == 0:
                return 1

            try:
                if str1 == str2:
                   simil = 1
                else:
                   simil = self.similarity_measure(str1, str2)

                if simil > 0:
                    weight = self.field_weights[field_name]
                    dist += (1/simil) * (weight/sum_weights)

            except Exception as exc:
                log.exception('Error calculating DICOM file distance.')

        if len(field_weights) == 0:
            return 1

        return dist


class SimpleDicomFileDistance(DicomFileDistance):

    def __init__(self, field_weights=DICOM_FIELD_WEIGHTS):
        super(SimpleDicomFileDistance, self).__init__(field_weights)

    def transform(self):
        """
        Checks the field values in self.dcmf1 and self.dcmf2 and returns True
        if all the field values are the same, False otherwise.

        :return: bool
        """
        if self.dcmf1 is None or self.dcmf2 is None:
            return np.inf

        for field_name in self.field_weights:
            if (str(getattr(self.dcmf1, field_name, ''))
                    != str(getattr(self.dcmf2, field_name, ''))):
                return False

        return True


def group_dicom_files(dicom_file_paths, header_fields):
    """
    Gets a list of DICOM file absolute paths and returns a list of lists of
    DICOM file paths. Each group contains a set of DICOM files that have
    exactly the same headers.

    Parameters
    ----------
    dicom_file_paths: list of str
    List or set of DICOM file paths

    header_fields: list of str
    List of header field names to check on the comparisons of the DICOM files.

    Returns
    -------
    dict of DicomFileSets
    """
    dist = SimpleDicomFileDistance()

    path_list = dicom_file_paths.copy()

    path_groups = DefaultOrderedDict(DicomFileSet)

    while len(path_list) > 0:
        file_path1 = path_list.pop()
        file_subgroup = [file_path1]

        dist.set_dicom_file1(file_path1)
        j = len(path_list)-1
        while j > 0:
            file_path2 = path_list[j]
            dist.set_dicom_file2(file_path2)

            if dist.transform():
                file_subgroup.append(file_path2)
                path_list.pop(j)

            j -= 1
        path_groups[file_path1].from_set(file_subgroup, check_if_dicoms=False)

    return path_groups


class DicomFilesClustering(object):
    """A self-organizing set of DICOM files.
    It uses macuto.dicom.comparison.DicomDistanceMeasure to compare
    all DICOM files within a set of folders and create clusters of DICOM files
    that are similar.

    This has been created to automatically generate sets of files for different
    subjects.

    Set the fields/weights in DICOM_FIELD_WEIGHTS to adjust this.
    """

    def __init__(self, folders=None, header_fields=None):
        """

        Parameters
        ----------
        folders: str or list of str
        Paths to folders containing DICOM files.
        If None, won't look for files anywhere.

        header_fields: dict or list of str to float
        Can be either a list of header field names that will be used for
        the SimpleDicomDistance and check str total similarity.
        Can also be a dict with header field names to float scalar values, that
        indicate a distance measure ratio for the levenshtein distance
        averaging of all the header field names in it. e.g., {'PatientID': 1}
        """
        self._file_dists = None
        self._subjs = DefaultOrderedDict(list)

        self._dicoms = DicomFileSet(folders)
        self.field_weights = header_fields

        if isinstance(header_fields, list) or isinstance(header_fields, tuple):
            self.headers = header_fields
        elif isinstance(header_fields, dict):
            self.headers = tuple(header_fields.keys())
        else:
            msg = 'header_fields parameter is neither list, tuple or dict'
            log.error(msg)
            raise ValueError(msg)

        self.dicom_groups = group_dicom_files(self._dicoms.items, self.headers)

    @staticmethod
    def calculate_file_distances(dicom_files, field_weights,
                                 dist_method=None):
        """
        Calculates the DicomFileDistance between all files in dicom_files,
        using an weighted Levenshtein measure between all field names in
        field_weights and their corresponding weights.

        Parameters
        ----------
        dicom_files: iterable of str
        Dicom file paths

        field_weights: dict of str to float
        A dict with header field names to float scalar values, that
        indicate a distance measure ratio for the levenshtein distance
        averaging of all the header field names in it. e.g., {'PatientID': 1}

        dist_method: DicomFileDistance object
        Distance method object to compare the files.
        If None, the default DicomFileDistance method using Levenshtein
        distance between the field_wieghts will be used.

        Returns
        -------
        file_dists: np.ndarray or scipy.sparse.lil_matrix of shape NxN
        Levenshtein distances between each of the N items in dicom_files.
        """
        if dist_method is None:
            log.info('Calculating Levenshtein distances between {0} DICOM '
                     'files.'.format(len(dicom_files)))
            dist_method = DicomFileDistance(field_weights)

        dist_dtype = np.float16
        n_files = len(dicom_files)

        try:
            file_dists = np.zeros((n_files, n_files), dtype=dist_dtype)
        except MemoryError as mee:
            import scipy.sparse
            file_dists = scipy.sparse.lil_matrix((n_files, n_files),
                                                 dtype=dist_dtype)

        for idxi in range(n_files):
            dist_method.set_dicom_file1(dicom_files[idxi])

            for idxj in range(idxi+1, n_files):
                dist_method.set_dicom_file2(dicom_files[idxj])

                if idxi != idxj:
                    file_dists[idxi, idxj] = dist_method.transform()

        return file_dists

    def levenshtein_analysis(self, field_weights=None):
        """
        Updates the status of the file clusters comparing the cluster
        key files with a levenshtein weighted measure using either the
        header_fields or self.header_fields.

        Parameters
        ----------

        field_weights: dict of strings with floats
        A dict with header field names to float scalar values, that
        indicate a distance measure ratio for the levenshtein distance
        averaging of all the header field names in it. e.g., {'PatientID': 1}
        """
        if field_weights is None:
            if not isinstance(self.field_weights, dict):
                msg = 'Expected a dict for field_weights parameter, ' \
                      'got {}'.format(type(self.field_weights))
                log.error(msg)
                raise ValueError(msg)

        key_dicoms = list(self.dicom_groups.keys())
        file_dists = self._calculate_file_distances(key_dicoms, field_weights)
        return file_dists

    @staticmethod
    def dist_percentile_threshold(dist_matrix, perc_thr=0.05, k=1):
        """Thresholds a distance matrix and returns the result.

        Parameters
        ----------

        dist_matrix: array_like
        Input array or object that can be converted to an array.

        perc_thr: float in range of [0,100]
        Percentile to compute which must be between 0 and 100 inclusive.

        k: int, optional
        Diagonal above which to zero elements.
        k = 0 (the default) is the main diagonal,
        k < 0 is below it and k > 0 is above.

        Returns
        -------
        array_like

        """
        triu_idx = np.triu_indices(dist_matrix.shape[0], k=k)
        upper = np.zeros_like(dist_matrix)
        upper[triu_idx] = dist_matrix[triu_idx] < \
                          np.percentile(dist_matrix[triu_idx], perc_thr)
        return upper

    def get_groups_in_same_folder(self, folder_depth=3):
        """
        Returns a list of 2-tuples with pairs of dicom groups that
        are in the same folder within given depth.

        Parameters
        ----------
        folder_depth: int
        Path depth to check for folder equality.

        Returns
        -------
        list of tuples of str
        """
        group_pairs = []
        key_dicoms = list(self.dicom_groups.keys())
        idx = len(key_dicoms)
        while idx > 0:
            group1 = key_dicoms.pop()
            dir_group1 = get_folder_subpath(group1, folder_depth)
            for group in key_dicoms:
                if group.startswith(dir_group1):
                    group_pairs.append((group1, group))
            idx -= 1

        return group_pairs

    @staticmethod
    def plot_file_distances(dist_matrix):
        """
        Plots dist_matrix

        Parameters
        ----------
        dist_matrix: np.ndarray
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.matshow(dist_matrix, interpolation='nearest',
                   cmap=plt.cm.get_cmap('PuBu'))

        #all_patients = np.unique([header.PatientName for header in self._dicoms])
        #ax.set_yticks(list(range(len(all_patients))))
        #ax.set_yticklabels(all_patients)

    def from_dicom_set(self, dicom_set):
        self._dicoms = dicom_set

    def merge_groups(self, indices):
        """Extend the lists within the DICOM groups dictionary.
        The indices will indicate which list have to be extended by which
        other list.

        Parameters
        ----------
        indices: list or tuple of 2 iterables of int, bot having the same len
             The indices of the lists that have to be merged, both iterables
             items will be read pair by pair, the first is the index to the
             list that will be extended with the list of the second index.
             The indices can be constructed with Numpy e.g.,
             indices = np.where(square_matrix)
        """
        from ..more_collections import merge_dict_of_lists

        try:
            merged = merge_dict_of_lists(self.dicom_groups, indices,
                                         pop_later=True, copy=True)
            self.dicom_groups = merged
        except IndexError as ie:
            log.exception('Index out of range to merge DICOM groups.')
            return None


if __name__ == '__main__':

    def test_DicomFileDistance():

        from macuto.dicom.comparison import DicomFileDistance

        dist = DicomFileDistance()

        datadir = '/home/alexandre/Projects/bcc/macuto/macuto/dicom'
        file1 = os.path.join(datadir, 'subj1_01.IMA')
        file2 = os.path.join(datadir, 'subj1_02.IMA')
        file3 = os.path.join(datadir, 'subj2_01.IMA')

        print(dist.fit_transform(file1, file1))
        print('...........................................')
        print(dist.fit_transform(file1, file2))
        print('...........................................')
        print(dist.fit_transform(file3, file2))


    def test_DicomFilesClustering():
        from macuto.dicom.comparison import (SimpleDicomFileDistance,
                                             DicomFilesClustering)

        from macuto.config import DICOM_FIELD_WEIGHTS
        import matplotlib.pyplot as plt

        #datadir = '/media/alexandre/cobre/santiago/test'
        #datadir = '/media/alexandre/cobre/santiago/raw'

        datadir = '/scratch/santiago'
        field_weights = DICOM_FIELD_WEIGHTS

        #datadir = '/scratch/santiago'
        #%time dcmclusters = DicomFilesClustering(datadir, field_weights)
        #CPU times: user 10min 14s, sys: 44.8 s, total: 10min 59s
        #Wall time: 12min 23s

        #datadir = '/scratch/santiago'
        #%time dcmclusters = DicomFilesClustering(datadir)
        #CPU times: user 4min 40s, sys: 27.7 s, total: 5min 8s
        #Wall time: 6min 15s

        #%time dcmclusters = DicomFilesClustering(datadir, field_weights)
        #CPU times: user 5h 8min 4s, sys: 25min 32s, total: 5h 33min 37s
        #Wall time: 6h 39min 2s

        #dcmclusters.

        #dcmclusters._calculate_file_distances()

        dcmgroups = DicomFilesClustering(datadir, field_weights)

        def levenshtein_thr_plot(dcmgroups, field_weights, threshold=0.05):
            dists = dcmgroups.levenshtein_analysis(field_weights)
            dists_thr = dcmgroups.dist_percentile_threshold(dists, threshold)
            dcmgroups.plot_file_distances(dists_thr)
            return dists, dists_thr

        file_dists, thr = levenshtein_thr_plot(dcmgroups, field_weights, 0.05)
        bday_dists, thr = levenshtein_thr_plot(dcmgroups, {'PatientBirthDate': 1}, 0.05)
        name_dists, thr = levenshtein_thr_plot(dcmgroups, {'PatientName': 1}, 0.10)


        fw = {'PatientName': 1}
        dist_method = SimpleDicomFileDistance(fw)
        key_dcms = list(dcmgroups.dicom_groups.keys())
        fdists = DicomFilesClustering.calculate_file_distances(key_dcms, fw,
                                                               dist_method)
        DicomFilesClustering.plot_file_distances(fdists)

        dcmgroups.merge_dict_of_lists(np.where(fdists))

        #def print_dcm_attributes(field_names, )
        indices = np.where(fdists)[0]
        for i in indices:
            print(DicomFile(key_dcms[i]).get_attributes(fw.keys()))

        #--------------------- test
        def print_group_names_from(self, dcmgroups, bin_dist_matrix):
            """
            Print in stdout the pair of group names for each True value in the
            binary distance matrix bin_dist_matrix

            :param bin_dist_matrix: np.ndarray
             array with shape NxN where N is the number of groups
            """
            key_dicoms = list(dcmgroups.dicom_groups.keys())
            for i, j in zip(*np.where(bin_dist_matrix)):
                print(key_dicoms[i] + ' and \n' + key_dicoms[j])
                print('\n')


        #import pickle
        #pickle.dump(dcmgroups, open('/home/alexandre/Desktop/dcmcluster.pickle', 'wb'))
        #dcmgroups = pickle.load(open('/home/alexandre/Desktop/dcmcluster.pickle', 'rb'))

        dm = dcmgroups.dicom_groups
        dcmgroups.plot_file_distances(dm.take(dm <= dm.mean()))
        #TODO

        #'PatientID'/'ProtocolName'
        #join experiment



