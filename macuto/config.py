
import logging
from collections import OrderedDict

#Known two-part file extensions
ALLOWED_EXTS = {'.gz': {'.nii'}}

LOG_LEVEL = logging.DEBUG

#Acceptable extensions for DICOM files
DICOM_FILE_EXTENSIONS = ['.IMA', '.DICOM', '.DCM']
DICOM_FILE_EXTENSIONS.extend([x.lower() for x in DICOM_FILE_EXTENSIONS])
OUTPUT_DICOM_EXTENSION = '.dcm'

#DICOM field weights for dicom.comparison
DICOM_FIELD_WEIGHTS = OrderedDict([('file_path', 0.25),
                                   ('PatientID', 1),
                                   ('PatientName', 1),
                                   ('PatientAddress', 0.5),
                                   ('PatientSex', 0.2),
                                   ('AcquisitionDate', 0.2),
                                   ('PatientBirthDate', 0.3)])