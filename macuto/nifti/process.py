__author__ = 'alexandre'

import os
from nipype.interfaces.fsl import IsotropicSmooth

from .nifti.

def smooth_volume(nifti_file, smoothmm):
    """

    @param nifti_file: string
    @param smoothmm: int
    @return:
    """
    if smoothmm > 0:
        omf = nifti_file + '_smooth' + str(smoothmm) + 'mm.nii.gz'

        isosmooth = IsotropicSmooth()
        isosmooth.inputs.in_file  = nifti_file
        isosmooth.inputs.fwhm     = smoothmm
        isosmooth.inputs.out_file = omf
        isosmooth.run()

        data = get_nii_data(omf)
        os.remove(omf)

    else:
        data = get_nii_data(nifti_file)

    return data