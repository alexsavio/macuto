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
The :mod:`macuto.timeseries` module includes functions to handle timeseries
from fMRI data: timeseries.selection has functions to select timeseries from
sets of fMRI timeseries, and timeseries.similarity_measure has functions to
measure similarities between sets of timeseries.
"""

__all__ = []


from .selection import (MeanTimeseries,
                        EigenTimeseries,
                        ILSIATimeseries,
                        CCATimeseries,
                        FilteredTimeseries,
                        MeanAndFilteredTimeseries,
                        EigenAndFilteredTimeseries,
                        )


from .similarity_measure import (TimeSeriesGroupMeasure,
                                 CorrelationMeasure,
                                 MeanCorrelationMeasure,
                                 CoherenceMeasure,
                                 MeanCoherenceMeasure,
                                 CrossCorrelationMeasure,
                                 )


from .connectivity import (create_connectivity_matrix,
                           select_timeseries,
                           calculate_similarities)

__sel_all__ = ['TimeseriesSelector',
               'MeanTimeseries',
               'EigenTimeseries',
               'ILSIATimeseries',
               'CCATimeseries',
               'FilteredTimeseries',
               'MeanAndFilteredTimeseries',
               'EigenAndFilteredTimeseries',]


__sm_all__ = ['SimilarityMeasure',
              'CorrelationMeasure',
              'MeanCorrelationMeasure',
              'CoherenceMeasure',
              'MeanCoherenceMeasure',
              'CrossCorrelationMeasure',]

__conn_all__ = ['create_connectivity_matrix',
                'select_timeseries',
                'calculate_similarities']

__all__.extend(__conn_all__)
__all__.extend(__sel_all__)
__all__.extend(__sm_all__)

