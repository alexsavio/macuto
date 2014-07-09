
from macuto.dicom.utils import DicomFile
import py.test

#def test_DicomFile():



if __name__ == '__main__':
    import os
    from macuto.dicom.utils import DicomFile
    datadir = '/home/alexandre/Projects/bcc/macuto/macuto/dicom'
    %timeit DicomFile(os.path.join(datadir, 'subj1_01.IMA'))