__author__ = 'alexandre'

import os
import dicom
import logging

from ..exceptions import LoggedError, FileNotFound
from ..config import DICOM_FIELD_WEIGHTS

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

    def fit(self, file_path1, file_path2):
        self.dcmf1 = DicomFile(file_path1)
        self.dcmf2 = DicomFile(file_path2)

    def fit_transform(self, file_path1, file_path2):
        self.fit(file_path1, file_path2)
        return self.transform()

    def transform(self):

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
