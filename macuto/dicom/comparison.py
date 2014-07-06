__author__ = 'alexandre'

import os
import logging
import numpy as np

from ..more_collections import DefaultOrderedDict
from ..config import DICOM_FIELD_WEIGHTS
from ..files.names import get_abspath
from ..exceptions import LoggedError
from .utils import DicomFile, get_dicom_file_paths

log = logging.getLogger(__name__)


class DistanceMeasure(object):
    #TODO
    #will be declared in ..classification.distance
    #currently from another branch
    pass


class DicomFileDistance(DistanceMeasure):

    import Levenshtein

    field_weights = DICOM_FIELD_WEIGHTS
    similarity_measure = Levenshtein.ratio
    inv_sum_weights = 1/sum(field_weights.values())

    def __init__(self):
        self.dcmf1 = None
        self.dcmf2 = None

    def fit(self, file_path1, file_path2):
        self.dcmf1 = DicomFile(file_path1)
        self.dcmf2 = DicomFile(file_path2)

    def fit_transform(self, file_path1, file_path2):
        self.fit(file_path1, file_path2)
        return self.transform()

    def transform(self):

        if self.dcmf1 is None or self.dcmf2 is None:
            return np.inf

        if len(self.field_weights) == 0:
            raise LoggedError('Field weights are not set.')

        try:
            dist = 0
            for field_name in self.field_weights:
                str1 = str(getattr(self.dcmf1, field_name))
                str2 = str(getattr(self.dcmf2, field_name))

                if not str1 or not str2:
                    continue

                weight = self.field_weights[field_name]
                simil = self.similarity_measure(str1, str2)

                if simil > 0:
                    dist += (1/simil) * weight

            return dist/len(self.field_weights) * self.inv_sum_weights

        except Exception as exc:
            raise LoggedError(str(exc))


class DicomFilesClustering(object):
    """A self-organizing set of DICOM files.
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
        self._file_dists = None
        self._subjs = DefaultOrderedDict(list)

        #self.dicom_files =
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

            self._files.extend(get_dicom_files_paths(folder))

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
        from macuto.dicom.sets import DicomFilesClustering
        wd = '/media/alexandre/cobre/santiago/data'
        dcmclusters = DicomFilesClustering(wd)
        #TODO