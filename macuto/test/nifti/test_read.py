import numpy as np
import nibabel as nib


def test_vector_to_volume():
    """
    This test is a piece of crap. Improve it!
    """
    img = nib.load('/usr/share/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
    vol = img.get_data()
    anat = nib.load('/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz')
    anatvol = anat.get_data()
    anatvol[np.where(vol > 0)]
    vec = anatvol[np.where(vol > 0)]
    from macuto.nifti.read import vector_to_volume
    vector_to_volume(vec, np.where(vol > 0), vol.shape)
