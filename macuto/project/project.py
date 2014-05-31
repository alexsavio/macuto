
from .config import Configuration
from .subject import Subject
from .clinical_data import ClinicalData


class Project(object):
    """
    Represent a study with subjects.
    """
    def __init__(self):
        self.config
        self.subjects #list of subjects
        self.clinics #pandas DataFrame

    #check data consistency (checks: if all subjects have the same data,
    #                                if all subjects have clinical and imaging data

    #anonimyze subjects (save and encrypt file with private data)
    #convert dicoms to nifti