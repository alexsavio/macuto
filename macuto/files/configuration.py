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


class Configuration(object):
    """
    Class to set dictionary keys as map attributes
    """
    def __init__(self, config_map):
        """
        :param config_map: dict
        """
        for key in config_map:
            if config_map[key] == 'None':
                config_map[key] = None
            setattr(self, key, config_map[key])


class Singleton(object):
    """
    Singleton class pattern
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        :param cls:
        :param args:
        :param kwargs:
        :return:
        """
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance