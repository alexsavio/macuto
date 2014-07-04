__author__ = 'alexandre'

import os
import dicom
from dicom.dataset import FileDataset

from ..exceptions import LoggedError, FileNotFound


FIELD_WEIGHTS = {'file_path': 0.4,
                 'PatientID': 1,
                 'PatientName': 1,
                 'PatientAddress': 1,
                 'PatientSex': 0.2,
                 'AcquisitionDate': 0.4,
                 'PatientBirthDate': 0.3}


class DistanceMeasure(object):
    #TODO
    #will be declared in ..classification.distance
    #currently from another branch
    pass


class DicomFile(FileDataset):

    def __init__(self, file_path, preamble=None, file_meta=None,
                 is_implicit_VR=True, is_little_endian=True):
        """

        :param file_path: full path and filename to the file.
        Use None if is a BytesIO.

        :param dataset: some form of dictionary, usually a Dataset from read_dataset()

        :param preamble: the 128-byte DICOM preamble
        :param file_meta: the file meta info dataset, as returned by _read_file_meta,
                or an empty dataset if no file meta information is in the file
        :param is_implicit_VR: True if implicit VR transfer syntax used; False if explicit VR. Default
 is True.
        :param is_little_endian: True if little-endian transfer syntax used; False if big-endian. Defa
ult is True.
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

    field_weights = FIELD_WEIGHTS
    distance_measure = Levenshtein.ratio

    def fit_transform(self, file_path1, file_path2):

        if len(self.field_weights) == 0:
            raise LoggedError('Field weights are not set.')

        dcm1 = DicomFile(file_path1)
        dcm2 = DicomFile(file_path2)

        try:
            dist = 0
            for field_name in self.field_weights:
                str1 = getattr(dcm1, field_name)
                str2 = getattr(dcm2, field_name)
                weight = self.field_weights[field_name]
                dist += self.distance_measure(str1, str2) * weight

            return dist/len(self.field_weights)

        except Exception as exc:
            raise LoggedError(str(exc))

