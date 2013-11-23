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


__version__ = '0.1-git'

__all__ = ['macuto', 'macuto.classification', 'macuto.nifti',]

from .io import(save_varlist_to_shelve,
                save_varlist_to_mat,
                save_varlist_to_hdf5,
                ExportData,
                load_variables_from_hdf5,
                load_varnames_from_hdf5,)

from .file_search import (find, 
                          get_file_list,
                          recursive_find,
                          iter_recursive_find,)

from .plot import (save_fig_to_png)

from .strings import (filter_objlist,
                      append_to_keys,
                      pretty_mapping)

from .files import (add_extension_if_needed,
                    get_extension,
                    remove_ext,
                    write_lines,
                    grep_one,
                    create_subjects_file,
                    parse_subjects_list)

__io_all__ = ['save_varlist_to_shelve',
              'save_varlist_to_mat',
              'save_varlist_to_hdf5',
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
                   'pretty_mapping']

__files_all__ = ['add_extension_if_needed',
                 'get_extension',
                 'remove_ext',
                 'write_lines',
                 'grep_one',
                 'create_subjects_file',
                 'parse_subjects_list']

__plot_all__ = ['save_fig_to_png']

__all__.extend(__io_all__)
__all__.extend(__plot_all__)
__all__.extend(__files_all__)
__all__.extend(__strings_all__)
__all__.extend(__file_search_all__)
