
import logging

#Known two-part file extensions
ALLOWED_EXTS = {'.gz': {'.nii'}}

LOG_LEVEL = logging.DEBUG

#Acceptable extensions for DICOM files
DICOM_FILE_EXTENSIONS = ['.IMA', '.DICOM', '.DCM']
DICOM_FILE_EXTENSIONS.extend([x.lower() for x in DICOM_FILE_EXTENSIONS])
OUTPUT_DICOM_EXTENSION = '.dcm'
