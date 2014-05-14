#!/usr/bin/env python

"""
Macuto
------

Macuto is a set of tools for management, processing and
statistical analysis for clinical and neuroimaging data.

"""
from __future__ import print_function

import os
import io
import sys
from setuptools import Command, setup, find_packages
from setuptools.command.test import test as TestCommand
from pip.req import parse_requirements

script_path = os.path.join('macuto', 'scripts')

install_reqs = parse_requirements('requirements.txt')

setup_dict = dict(
    name = 'macuto',
    version = '0.2.0',
    description = 'MRI Analysis and Classification Tools',

    license = 'BSD 3-Clause',
    author = 'Alexandre M. Savio',
    author_email = 'alexsavio@gmail.com',
    maintainer = 'Alexandre M. Savio',
    maintainer_email = 'alexsavio@gmail.com',

    #packages = ['macuto', 'macuto.nifti',
    #            'macuto.classification', 'macuto.timeseries',
    #            'macuto.atlas']
    packages = find_packages(),

    #install_requires = ['numpy', 'scipy', 'scikit-learn', 'matplotlib', 
    #                    'nibabel', 'h5py', 'nitime']
    install_requires = [str(ir.req) for ir in install_reqs],

    extra_files = ['CHANGES.rst', 'LICENSE', 'README.rst'],

    scripts = ['macuto/scripts/dcm_anonymize.py',
               'macuto/scripts/sav_convert.py',
               'macuto/scripts/filetree.py',
               'macuto/scripts/sliceit.py'],

    platforms='any',

    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Information Analysis',
        ],

    extras_require={
        'testing': ['pytest'],
    }
)


#long description
def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


setup_dict.update(dict(long_description = read('README.rst', 'CHANGES.rst')))


#Python3 support keywords
if sys.version_info >= (3,):
    setup_dict['use_2to3'] = False
    setup_dict['convert_2to3_doctests'] = ['']
    setup_dict['use_2to3_fixers'] = ['']


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup_dict.update(dict(tests_require=['pytest'],
                       cmdclass={'test': PyTest}))


class run_audit(Command):
    """Audits source code using PyFlakes for following issues:
- Names which are used but not defined or used before they are defined.
- Names which are redefined without having been used.
"""
    description = "Audit source code with PyFlakes"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import os
        import sys
        try:
            import pyflakes.scripts.pyflakes as flakes
        except ImportError:
            print("Audit requires PyFlakes installed in your system.")
            sys.exit(-1)

        warns = 0
        # Define top-level directories
        dirs = ('macuto')
        for dir in dirs:
            for root, _, files in os.walk(dir):
                for file in files:
                    if file != '__init__.py' and file.endswith('.py'):
                        warns += flakes.checkPath(os.path.join(root, file))
        if warns > 0:
            print("Audit finished with total %d warnings." % warns)
        else:
            print("No problems found in sourcecode.")


if __name__ == '__main__':
    setup(**setup_dict)
