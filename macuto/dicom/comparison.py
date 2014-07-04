__author__ = 'alexandre'

import os
import dicom
from dicom.dataset import FileDataset
import logging

from ..exceptions import LoggedError, FileNotFound
from ..config import DICOM_FIELD_WEIGHTS

log = logging.getLogger(__name__)


class DistanceMeasure(object):
    #TODO
    #will be declared in ..classification.distance
    #currently from another branch
    pass


class DicomFile(FileDataset):

    def __init__(self, file_path, preamble=None, file_meta=None,
                 is_implicit_VR=True, is_little_endian=True):
        """

        :param file_path: str
        Full path and filename to the file.
        Use None if is a BytesIO.

        :param dataset:
        some form of dictionary, usually a Dataset from read_dataset()

        :param preamble: the 128-byte DICOM preamble

        :param file_meta: dataset
        The file meta info dataset, as returned by _read_file_meta,
        or an empty dataset if no file meta information is in the file

        :param is_implicit_VR: bool
        True if implicit VR transfer syntax used; False if explicit VR.
        Default is True.

        :param is_little_endian: bool
         True if little-endian transfer syntax used; False if big-endian.
         Default is True.
        """
        if not os.path.exists(file_path):
            raise FileNotFound(file_path)

        try:
            dcm = dicom.read_file(file_path)

            FileDataset.__init__(self, file_path, dcm, preamble, file_meta,
                                 is_implicit_VR, is_little_endian)

            self.file_path = os.path.abspath(file_path)
        except Exception as exc:
            raise LoggedError(str(exc))


class DicomFileDistance(DistanceMeasure):

    import Levenshtein

    field_weights = DICOM_FIELD_WEIGHTS
    similarity_measure = Levenshtein.ratio
    inv_sum_weights = 1/sum(field_weights.values())

    def fit_transform(self, file_path1, file_path2):

        if len(self.field_weights) == 0:
            raise LoggedError('Field weights are not set.')

        dcm1 = DicomFile(file_path1)
        dcm2 = DicomFile(file_path2)

        try:
            dist = 0
            for field_name in self.field_weights:
                str1 = str(getattr(dcm1, field_name))
                str2 = str(getattr(dcm2, field_name))

                if not str1 or not str2:
                    continue

                weight = self.field_weights[field_name]
                simil = self.similarity_measure(str1, str2)

                if simil > 0:
                    dist += (1/simil) * weight

            return dist/len(self.field_weights) * self.inv_sum_weights

        except Exception as exc:
            raise LoggedError(str(exc))


if __name__ == '__main__':
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
