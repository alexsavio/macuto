
import logging
import numpy as np
from nipy.algorithms.kernel_smooth import LinearFilter

from .read import load_nipy_img
from ..files.names import get_abspath
from ..more_collections import ItemSet

log = logging.getLogger(__name__)


class NiftiSubjectsSet(ItemSet):

    def __init__(self, subj_files, mask_file=None, all_same_shape=True):
        """

        :param subj_files: list or dict of str
        file_path -> int/str

        :param mask_file: str

        :param all_same_size: bool
        True if all the subject files must have the same shape
        """
        self.items = []
        self.labels = []
        self.all_same_shape = all_same_shape

        self.mask_file = mask_file
        self._init_subj_data(subj_files)

        if all_same_shape:
            self._check_subj_shapes()

    def _init_subj_data(self, subj_files):
        """

        :param subj_files:
        :return:
        """
        try:
            if isinstance(subj_files, list):
                self.from_list(subj_files)
            elif isinstance(subj_files, dict):
                self.from_dict(subj_files)
            else:
                raise ValueError('Could not recognize subj_files argument '
                                 'variable type.')
        except Exception as exc:
            log.exception('Cannot read subj_files input argument')

    def _check_subj_shapes(self):
        """

        :return:
        """
        shape = self.items[0].shape

        for img in self.items:
            if img.shape != shape:
                raise ValueError('Shape mismatch in '
                                 'file {0}.'.format(img.file_path))

    @staticmethod
    def _load_image(file_path):
        """

        :param file_path: str
         Path to the nifti file

        :return: nipy.Image with a file_path member
        """
        try:
            nii_img = load_nipy_img(file_path)
            nii_img.file_path = file_path
            return nii_img
        except Exception as exc:
            log.exception('Reading file {0}.'.format(file_path))

    @staticmethod
    def _smooth_img(nii_img, smooth_mm):
        """
        """
        if smooth_mm <= 0:
            return nii_img

        filter = LinearFilter(nii_img.coordmap, nii_img.shape)
        return filter.smooth(nii_img)

    def from_dict(self, subj_files):
        """

        :param subj_files:
        """
        for group_label in subj_files:
            try:
                group_files = subj_files[group_label]
                self.items.extend([self._load_image(get_abspath(imgf))
                                   for imgf in group_files])

                self.labels.append(group_label)

            except Exception as exc:
                log.exception('Error while reading files from '
                              'group {0}.'.format(group_label))

    def from_list(self, subj_files):
        """

        :param subj_files:
        :return:
        """
        for sf in subj_files:
            try:
                nii_img = self._load_image(get_abspath(sf))
                self.items.append(nii_img)
            except Exception as exc:
                log.exception('Error while reading file {0}.'.format(sf))

    @property
    def n_subjs(self):
        return len(self.items)

    @property
    def has_mask(self):
        return self.mask_file is not None

    def set_labels(self, subj_labels):
        """

        :param subj_labels: list of int or str
         This list will be checked to have the same size as files list
         (self.items)
        """
        if len(subj_labels) != self.n_subjs:
            log.error('The number of given labels is not the same '
                      'as the number of subjects.')

        self.labels = subj_labels

    def to_matrix(self, smooth_mm, smooth_mask=False, outdtype=None):
        """
        Creates a Numpy array with the data.

        :param smooth__mm: int
        Integer indicating the size of the FWHM Gaussian smoothing kernel
        to smooth the subject volumes before creating the data matrix

        :param smooth_mask: bool
        If True, will smooth the mask with the same kernel.

        :param outdtype: dtype
        Type of the elements of the array, if None will obtain the dtype from
        the first nifti file.

        Returns:
        --------
        :return: outmat, vol_shape,

        outmat: Numpy array with shape N x prod(vol.shape)
                containing the N files as flat vectors.

        mask_indices: matrix with indices of the voxels in the mask

        vol_shape: Tuple with shape of the volumes, for reshaping.
        """

        vol = self.items[0].get_data()
        if not outdtype:
            outdtype = vol.dtype

        n_voxels = None
        mask_indices = None
        mask_shape = None

        if self.has_mask:
            mask = load_nipy_img(self.mask_file)

            if smooth_mask:
                mask = self._smooth_img(mask, smooth_mm)

            mask = mask.get_data()
            mask_indices = np.where(mask > 0)
            mask_shape = mask.shape
            n_voxels = np.count_nonzero(mask)

        if n_voxels is None:
            n_voxels = np.prod(vol.shape)

        outmat = np.zeros((self.n_subjs, n_voxels), dtype=outdtype)
        try:
            for i, vf in enumerate(self.items):
                vol = self._smooth_img(vf, smooth_mm).get_data()
                if mask_indices is not None:
                    outmat[i, :] = vol[mask_indices]
                else:
                    outmat[i, :] = vol.flatten()
        except Exception as exc:
            log.exception('Flattening file {0}'.format(vf.file_path))

        return outmat, mask_indices, mask_shape

if __name__ == '__main__':
    #NiftiSubjectsSet(subj_files, mask_file=None, all_same_shape=True)
    pass
