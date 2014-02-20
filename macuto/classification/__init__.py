# coding=utf-8
#-------------------------------------------------------------------------------

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

from .classification import (perform_classification)


from .sklearn import (get_clfmethod,
                      get_cv_method,
                      get_fsmethod,
                      get_pipeline)


from .results import (Result,
                      classification_metrics,
                      enlist_cv_results,
                      get_confusion_matrix_fisher_significance,
                      get_cv_classification_metrics,
                      get_cv_significance)

from .features import (calculate_hist3d,
                       create_feature_sets,
                       calculate_stats,
                       distance_computation,
                       bhattacharyya_dist,
                       feature_selection,
                       pearson_correlation,
                       welch_ttest)

from .plot import (save_fig_to_png,
                   plot_results)

__all_skl__ = ['get_clfmethod',
               'get_cv_method',
               'get_fsmethod',
               'get_pipeline']


__all_io__ = ['parse_subjects_list',
              'load_data',
              'write_arff',
              'write_svmperf_dat']


__all_rslts__ = ['Result',
                 'classification_metrics',
                 'enlist_cv_results',
                 'get_confusion_matrix_fisher_significance',
                 'get_cv_classification_metrics',
                 'get_cv_significance']


__all_feats__ = ['calculate_hist3d',
                 'create_feature_sets',
                 'calculate_stats',
                 'distance_computation',
                 'bhattacharyya_dist',
                 'feature_selection',
                 'pearson_correlation',
                 'welch_ttest']

__all_plot__ = ['save_fig_to_png',
                'plot_results']

__all__.extend(__all_io__)
__all__.extend(__all_skl__)
__all__.extend(__all_feats__)
__all__.extend(__all_rslts__)
__all__.extend(__all_plot__)
