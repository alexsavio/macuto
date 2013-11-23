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
The :mod:`macuto.nifti` module includes functions for Nifti and Nipy files
handling/processing
"""

__all__ = []

from .nifti_data_import import (get_nii_data,
                                get_masked_nii_data,
                                get_nii_info,
                                niftilist_mask_to_array,
                                niftilist_to_array,
                                smooth_volume)

from .image_info import (is_valid_coordinate,
                         are_compatible_imgs,
                         check_have_same_geometry,
                         have_same_geometry,
                         have_same_shapes)

from .coord_transform import (voxcoord_to_mm,
                              mm_to_voxcoord,
                              get_3D_coordmap,
                              get_coordmap_array)

from .io import (spatialimg_to_hdf,
                 hdfgroup_to_nifti1image,
                 get_nifti1hdr_from_h5attrs)

from .roi import (drain_rois)

__ndi_all__ = ['get_nii_data',
               'get_masked_nii_data',
               'get_nii_info',
               'niftilist_mask_to_array',
               'niftilist_to_array',
               'smooth_volume']

__ii_all__ = ['is_valid_coordinate',
              'are_compatible_imgs',
              'check_have_same_geometry',
              'have_same_geometry',
              'have_same_shapes']

__ct_all__ = ['voxcoord_to_mm',
              'mm_to_voxcoord',
              'get_3D_coordmap',
              'get_coordmap_array']

__io_all__ = ['spatialimg_to_hdf',
              'hdfgroup_to_nifti1image',
              'get_nifti1hdr_from_h5attrs']

__roi_all__ = ['drain_rois']

__all__.extend(__ndi_all__)
__all__.extend(__roi_all__)
__all__.extend(__ii_all__)
__all__.extend(__ct_all__)
__all__.extend(__io_all__)