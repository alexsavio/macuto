#MR Automatic Classification TOols

import os
from os.path import join
import warnings


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    config = Configuration('macuto', parent_package, top_path)

    config.add_subpackage('io')
    config.add_subpackage('nifti')
    config.add_subpackage('classification')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(name='macuto',
      description='MR Automatic Classification TOols',
      author='Alexandre Manhaes Savio',
      author_email='alexsavio@gmail.com',
      url='http://www.github.com/alexsavio/macuto/',
      **configuration(top_path='').todict()
      )

    
