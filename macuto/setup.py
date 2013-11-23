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
MR Automatic Classification TOols (macuto)
"""


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('macuto', parent_package, top_path)

    config.add_subpackage('nifti')
    config.add_subpackage('classification')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(name='macuto',
          description='MR Automatic Classification TOols (macuto)',
          author='Alexandre Manhaes Savio',
          author_email='alexsavio@gmail.com',
          url='http://www.github.com/alexsavio/macuto/',
          **configuration(top_path='').todict()
          )
