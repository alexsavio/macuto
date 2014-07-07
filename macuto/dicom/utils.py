import os
import dicom
import logging
import subprocess
from collections import defaultdict
from dicom.dataset import FileDataset

from ..exceptions import LoggedError, FileNotFound

log = logging.getLogger(__name__)


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


def get_dicom_file_paths(dirpath):
    return [os.path.join(dp, f) for dp, dn, filenames in
            os.walk(dirpath) for f in filenames
            if is_dicom_file(os.path.join(dp, f))]


def get_dicomfiles(dirpath):
    return [DicomFile(os.path.join(dp, f)) for dp, dn, filenames in
            os.walk(dirpath) for f in filenames
            if is_dicom_file(os.path.join(dp, f))]


def is_dicom_file(filepath):
    """
    Tries to read the file using dicom.read_file,
    if the file exists and dicom.read_file does not raise
    and Exception returns True. False otherwise.

    :param filepath: str
     Path to DICOM file

    :return: bool
    """
    if not os.path.exists(filepath):
        return False

    filename = os.path.basename(filepath)
    if filename == 'DICOMDIR':
        return False

    try:
        _ = dicom.read_file(filepath)
    except Exception as exc:
        LoggedError(str(exc))
        return False

    return True


def group_dicom_files(dicom_paths, hdr_field='PatientID'):
    """

    :param dicom_paths: str
    :return: dict of dicom_paths
    """
    dicom_groups = defaultdict(list)
    for dcm in dicom_paths:
        hdr = dicom.read_file(dcm)

        try:
            group_key = getattr(hdr, hdr_field)
        except:
            raise

        dicom_groups[group_key].append(dcm)

    return dicom_groups


def call_dcm2nii(input_path):
    """

    :param input_path: str
    :return:
    """
    try:
        log.info('dcm2nii {0}'.format(input_path))
        return subprocess.call('dcm2nii {0}'.format(input_path),
                               shell=True)

    except Exception as e:
        raise LoggedError('Error calling dcm2nii on {0}. {1}'.format(input_path,
                                                                     str(e)))


def anonymize_dicom_file(dcm_file, remove_private_tags=False,
                         remove_curves=False):
    """Anonymizes the given dcm_file.

    Anonymizing means: putting nonsense information into tags:
    PatientName, PatientAddress and PatientBirthDate.

    :param acqfolder: path.py path
    Path to the DICOM file.
    """
    assert(dcm_file.isfile())

    # Load the current dicom file to 'anonymize'
    plan = dicom.read_file(dcm_file)

    plan.PatientName = 'Anonymous'
    plan.PatientAddress = 'North Pole'

    # Define call-back functions for the dataset.walk() function
    def PN_callback(ds, data_element):
        """Called from the dataset "walk" recursive function for all data elements."""
        if data_element.VR == "PN":
            data_element.value = 'Anonymous'

    def curves_callback(ds, data_element):
        """Called from the dataset "walk" recursive function for all data elements."""
        if data_element.tag.group & 0xFF00 == 0x5000:
            del ds[data_element.tag]

    # Remove patient name and any other person names
    plan.walk(PN_callback)

    # Remove data elements (should only do so if DICOM type 3 optional)
    # Use general loop so easy to add more later
    # Could also have done: del ds.OtherPatientIDs, etc.
    #for name in ['OtherPatientIDs']:
    #    if name in plan:
    #        delattr(ds, name)

    # Same as above but for blanking data elements that are type 2.
    for name in ['PatientsBirthDate']:
        if name in plan:
            plan.data_element(name).value = ''

    # Remove private tags if function argument says to do so. Same for curves
    if remove_private_tags:
        plan.remove_private_tags()
    if remove_curves:
        plan.walk(curves_callback)

    # write the 'anonymized' DICOM out under the new filename
    plan.save_as(dcm_file)


def anonymize_dicom_file_dcmtk(dcm_file):
    """Anonymizes the given dcm_file.

    Anonymizing means: putting nonsense information into tags:
    PatientName, PatientAddress and PatientBirthDate.

    :param acqfolder: path.py path
    Path to the DICOM file.
    """
    assert(dcm_file.isfile())

    subprocess.call('dcmodify --modify PatientName=Anonymous ' + dcm_file,
                    shell=True)
    subprocess.call('dcmodify --modify PatientBirthDate=17000101 ' + dcm_file,
                    shell=True)
    subprocess.call('dcmodify --modify PatientAddress=North Pole ' + dcm_file,
                    shell=True)

    os.remove(dcm_file + '.bak')

if __name__ == '__main__':

    from macuto.dicom.utils import DicomFile

    dcm_file_hd = '/home/alexandre/Projects/bcc/macuto/macuto/dicom/subj1_01.IMA'
    #%timeit DicomFile(dcm_file_hd)
    #1000 loops, best of 3: 1.75 ms per loop

    dcm_file_ssd = '/scratch/subj1_01.IMA'
    #%timeit DicomFile(dcm_file_ssd)
    #1000 loops, best of 3: 1.75 ms per loop