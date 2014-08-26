__author__ = 'alexandre'

import os
import logging
import subprocess

from ..nifti.storage import save_niigz
from ..files.names import get_temp_file, get_temp_dir

log = logging.getLogger(__name__)


class FslViewCaller(object):

    fslview_bin = os.path.join(os.environ['FSLDIR'], 'bin', 'fslview')

    def __init__(self):
        self._tmpdir = get_temp_dir('fslviewcaller_')
        self._volume_files = set()
        self._tmp_volume_files = set()

    def add_volume_from_path(self, nii_path):
        if not os.path.exists(nii_path):
            log.error('File {} not found.'.format(nii_path))

        self._add_volume_from_path(nii_path, is_tmp_file=False)

    def add_volume(self, vol_data, affine=None, header=None):
        tmp_file = get_temp_file(self._tmpdir.name, suffix='.nii.gz')
        save_niigz(tmp_file.name, vol_data, affine, header)
        self._add_volume_from_path(tmp_file, is_tmp_file=True)

    def _add_volume_from_path(self, nii_path, is_tmp_file):
        if is_tmp_file:
            self._tmp_volume_files.add(nii_path)
        else:
            self._volume_files.add(nii_path)

    def show(self):
        fslview_args = [self.fslview_bin]
        fslview_args.extend(self._volume_files)
        fslview_args.extend(self._tmp_volume_files)
        subprocess.call(fslview_args)

    def close(self):
        import shutil
        try:
            for volf in self._tmp_volume_files:
                os.remove(volf)
        except OSError:
            log.exception('Error closing {} on deleting '
                          'file {}.'.format(self.__name__, volf))
            raise

        try:
            shutil.rmtree(self._tmpdir.name)
        except OSError:
            log.exception('Error closing {} on deleting '
                          'temp folder {}.'.format(self._tmpdir))
