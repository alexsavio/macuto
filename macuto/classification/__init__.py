# coding=utf-8
#-------------------------------------------------------------------------------
#License GNU/GPL v3
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

"""
The :mod:`macuto.io` module includes input/output and file handling functions
"""

__all__ = []

from .io import (parse_subjects_list,
                 load_data,
                 write_arff,
                 write_svmperf_dat)

from .classification import (perform_classification)

from .sklearn import ()

from .results import ()

from .threshold import (apply__threshold,
                        binarise,
                        find_histogram,
                        find_thresholds,
                        is_symmetric,
                        percentile_threshold,
                        rank_threshold,
                        robust_max,
                        robust_min,
                        robust_range_threshold,
                        threshold,
                        threshold_robust_range)

__all_io__ = ['parse_subjects_list',
              'load_data',
              'write_arff',
              'write_svmperf_dat']

__all__.extend(__all_io__)

