import os
import dicom
import logging
import subprocess
from collections import defaultdict
from dicom.dataset import FileDataset

import macuto.files.search as fs

log = logging.getLogger(__name__)


class DicomFile(FileDataset):
    """Store the contents of a DICOM file

    Parameters
    ----------
    file_path: str
     Full path and filename to the file.
     Use None if is a BytesIO.

    header_fields: subset of DICOM header fields to be
     stored here, the rest will be ignored.

    dataset: dict
     Some form of dictionary, usually a Dataset from read_dataset()

    preamble: the 128-byte DICOM preamble

    file_meta: dataset
     The file meta info dataset, as returned by _read_file_meta,
     or an empty dataset if no file meta information is in the file

    is_implicit_VR: bool
     True if implicit VR transfer syntax used; False if explicit VR.
     Default is True.

    is_little_endian: bool
     True if little-endian transfer syntax used; False if big-endian.
     Default is True.
    """
    def __init__(self, file_path, preamble=None, file_meta=None,
                 is_implicit_VR=True, is_little_endian=True):
        try:
            dcm = dicom.read_file(file_path, force=True)

            FileDataset.__init__(self, file_path, dcm, preamble, file_meta,
                                 is_implicit_VR, is_little_endian)

            self.file_path = os.path.abspath(file_path)

        except Exception as exc:
            log.exception('Error reading file {0}.'.format(file_path))
            raise

    def get_attributes(self, attributes, default=''):
        """Return the attributes values from this DicomFile

        Parameters
        ----------
        attributes: str or list of str
         DICOM field names

        default: str
         Default value if the attribute does not exist.

        Returns
        -------
        Value of the field or list of values.
        """
        if isinstance(attributes, str):
            attributes = [attributes]

        attrs = [getattr(self, attr, default) for attr in attributes]

        if len(attrs) == 1:
            return attrs[0]

        return tuple(attrs)


def get_dicom_files(dirpath):
    return [DicomFile(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(dirpath)
            for f in filenames if is_dicom_file(os.path.join(dp, f))]


def get_unique_field_values(dcm_file_list, field_name):
    """Return a set of unique field values from a list of DICOM files

    Parameters
    ----------
    dcm_file_list: iterable of DICOM file paths

    field_name: str
     Name of the field from where to get each value

    Returns
    -------
    Set of field values
    """
    field_values = set()
    try:
        for dcm in dcm_file_list:
            field_values.add(str(DicomFile(dcm).get_attributes(field_name)))
        return field_values
    except Exception:
        log.exception('Error reading file {}'.format(dcm))
        raise


def find_all_dicom_files(root_path):
    """
    Returns a list of the dicom files within root_path

    Parameters
    ----------
    root_path: str
    Path to the directory to be recursively searched for DICOM files.

    Returns
    -------
    dicoms: set
    Set of DICOM absolute file paths
    """
    dicoms = set()
    f = None
    try:
        for fpath in fs.get_all_files(root_path):
            if is_dicom_file(fpath):
                dicoms.add(fpath)
    except Exception as exc:
        log.exceptions('Error reading file {0}.'.format(fpath))

    return dicoms


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
        log.debug('Checking if {0} was a DICOM, but returned '
                  'False.'.format(filepath))
        return False

    return True


def group_dicom_files(dicom_paths, hdr_field='PatientID'):
    """

    :param dicom_paths: str
    :return: dict of dicom_paths
    """
    dicom_groups = defaultdict(list)
    try:
        for dcm in dicom_paths:
            hdr = dicom.read_file(dcm)
            group_key = getattr(hdr, hdr_field)
            dicom_groups[group_key].append(dcm)
    except Exception as exc:
        log.exception('Error reading file {0}.'.format(dcm))

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
        log.exception('Error calling dcm2nii on {0}.'.format(input_path))


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
