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
MR Automatic Classification TOols
==================================

macuto is a Python module integrating a series of other modules
for typical data import/export and classification steps.
It also includes typical Nifti brain image processing functions.

It aims towards a library for classification and data analysis
of different kinds of data.
We also believe that leveraging the complexities of different Python modules
into one library that performs specific and good methodological steps 
in classification and data analysis could bring non-technical users to 
the Python community.
"""


__all__ = ['macuto', 'macuto.classification', 'macuto.nifti',
           'macuto.timeseries', 'macuto.atlas']

from .io import(save_variables_to_hdf5,
                save_variables_to_mat,
                save_variables_to_shelve,
                ExportData,
                load_variables_from_hdf5,
                load_varnames_from_hdf5,)

from .file_search import (get_file_list,
                          recursive_find,
                          iter_recursive_find,)

from .plot import (save_fig_to_png,
                   plot_results,
                   subplot_this)

from .strings import (filter_objlist,
                      append_to_keys,
                      pretty_mapping,
                      list_filter,
                      list_search,
                      append_to_list,
                      list_match)

from .files import (add_extension_if_needed,
                    get_extension,
                    remove_ext,
                    write_lines,
                    grep_one,
                    create_subjects_file,
                    parse_subjects_list,
                    join_path_to_filelist,
                    remove_all,
                    get_temp_file,
                    count_lines,
                    file_size,
                    fileobj_size,
                    ux_file_len)

from .render import (show_3slices,
                     show_contour,
                     show_cutplanes,
                     show_dynplane,
                     show_many_slices,
                     show_render,
                     autocrop_img,
                     borders,
                     create_imglist_html,
                     imshow,
                     show_connectivity_matrix,
                     slicesdir_connectivity_matrices,
                     slicesdir_paired_overlays)

from .math import (makespread,
                   takespread)

from .threshold import (apply_threshold,
                        binarise,
                        find_histogram,
                        find_thresholds,
                        is_symmetric,
                        percentile_threshold,
                        rank_threshold,
                        robust_max,
                        robust_min,
                        robust_range_threshold,
                        threshold)

from .commands import (condor_call,
                       condor_submit)

from .databuffer import (HdfDataBuffer,
                         NumpyHDFStore)

__io_all__ = ['save_variables_to_shelve',
              'save_variables_to_mat',
              'save_variables_to_hdf5',
              'ExportData',
              'load_varnames_from_hdf5',
              'load_variables_from_hdf5']

__file_search_all__ = ['find', 
                       'get_file_list',
                       'recursive_find',
                       'iter_recursive_find',
                       'filter_objlist']

__strings_all__ = ['filter_objlist',
                   'append_to_keys',
                   'pretty_mapping',
                   'list_filter',
                   'list_search',
                   'append_to_list',
                   'list_match']

__files_all__ = ['add_extension_if_needed',
                 'get_extension',
                 'remove_ext',
                 'write_lines',
                 'grep_one',
                 'create_subjects_file',
                 'parse_subjects_list',
                 'join_path_to_filelist',
                 'remove_all',
                 'get_temp_file',
                 'count_lines',
                 'file_size',
                 'fileobj_size',
                 'ux_file_len']

__math_all__ = ['makespread',
                'takespread']

__render_all__ = ['show_3slices',
                  'show_contour',
                  'show_cutplanes',
                  'show_dynplane',
                  'show_many_slices',
                  'show_render',
                  'autocrop_img',
                  'borders',
                  'create_imglist_html',
                  'imshow',
                  'show_connectivity_matrix',
                  'slicesdir_connectivity_matrices',
                  'slicesdir_paired_overlays',]

__plot_all__ = ['save_fig_to_png',
                'plot_results',
                'subplot_this']

__thresh_all__ = ['apply__threshold',
                  'binarise',
                  'find_histogram',
                  'find_thresholds',
                  'is_symmetric',
                  'percentile_threshold',
                  'rank_threshold',
                  'robust_max',
                  'robust_min',
                  'robust_range_threshold',
                  'threshold']

__cmds_all__ = ['condor_call',
                'condor_submit']

__dbf_all__ = ['HdfDataBuffer',
               'NumpyHDFStore']

__all__.extend(__io_all__)
__all__.extend(__dbf_all__)
__all__.extend(__plot_all__)
__all__.extend(__math_all__)
__all__.extend(__cmds_all__)
__all__.extend(__files_all__)
__all__.extend(__render_all__)
__all__.extend(__thresh_all__)
__all__.extend(__strings_all__)
__all__.extend(__file_search_all__)
