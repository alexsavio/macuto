# coding=utf-8
#-------------------------------------------------------------------------------

#LICENSE: BSD 3-Clause
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2014, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

#Take into account:
#http://nipy.sourceforge.net/nipype/users/select_files.html

class FileManager(object):
    """
    This class is a Singleton registry of file lists.
    It works as a dictionary of file lists whose key is a given string, but
    giving further functionality related to relative paths, file indexing
    and searching.
    """
    _instance = None

    def __init__(self):
        """

        :return:
        """



    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance

    def