
from macuto.config import DICOM_FIELD_WEIGHTS
from macuto.dicom.sets import GenericDicomsList
from macuto.exceptions import LoggedError

datadir_hd = '/media/alexandre/cobre/santiago/test' #HD 4.2GB in 9981 DICOMS
#%timeit dicoms = DicomFileList(datadir_hd, store_metadata=True)
#1 loops, best of 3: 38.4 s per loop

datadir_ssd = '/scratch/santiago_test' #SSD 4.2GB in 9981 DICOMS
#%timeit dicoms = DicomFileList(datadir_ssd, store_metadata=True)
#1 loops, best of 3: 38 s per loop

datadir = '/scratch/santiago_test'
header_fields = tuple(DICOM_FIELD_WEIGHTS.keys())

def test_DicomHeader():

    import os
    from collections import namedtuple
    from macuto.dicom.utils import DicomFile
    from macuto.dicom.utils import is_dicom_file

    def _get_dicoms(build_dcm, root_path, header_fields=None):
        #return [build_dcm(dp, f, header_fields) for dp, dn, filenames in os.walk(root_path)
        #        for f in filenames if is_dicom_file(os.path.join(dp, f))]

        dicoms = []
        f = None
        try:
            for dp, dn, filenames in os.walk(root_path):
                for f in filenames:
                    fpath = os.path.join(dp, f)
                    if is_dicom_file(fpath):
                        dicoms.append(build_dcm(fpath, header_fields))
        except Exception as exc:
            raise LoggedError('Error reading file {0}. '
                              '{1}'.format(os.path.join(dp, f), str(exc)))

        return dicoms

    folder = datadir

    DicomHeader = namedtuple('DicomHeader', header_fields)

    build_dcm = lambda fpath, flds: DicomHeader._make(DicomFile(fpath).get_attributes(header_fields))

    dicoms = _get_dicoms(build_dcm, folder,  header_fields)


def test_GenericDicomsList():

    dicoms = GenericDicomsList(datadir, store_metadata=True,
                               header_fields=header_fields)


def test_GenericDicomsList2():
    dicoms = GenericDicomsList(datadir, store_metadata=False)
