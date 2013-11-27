#MR Automatic Classification TOols

descr = """A set of python modules for:
        - Generic file management
        - Nifti image managing with nibabel, h5py
        - Classification with scikit-learn
        - Feature extraction and job pipelining with scikit-learn
        - Visualization with matplotlib and mayavi
        - Plotting with matplotlib
        """

import sys
import os
import shutil
from distutils.command.clean import clean as Clean

if sys.version_info[0] < 3:
    import builtins as builtins
else:
    import builtins

DISTNAME = 'macuto'
DESCRIPTION = 'A set of python modules for MR Classification'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Alexandre M. Savio'
MAINTAINER_EMAIL = 'alexsavio@gmail.com'
URL = 'http://amsivo.com'
LICENSE = 'GPLv3'
DOWNLOAD_URL = 'http://sourceforge.net/projects/scikit-learn/files/'

# We can actually import a restricted version of sklearn that
# does not need the compiled code
import macuto
VERSION = macuto.__version__

###############################################################################
# Optional setuptools features
# We need to import setuptools early, if we want setuptools features,
# as it monkey-patches the 'setup' function

# For some commands, use setuptools
if len(set(('develop', 'release', 'bdist_egg', 'bdist_rpm',
           'bdist_wininst', 'install_egg_info', 'build_sphinx',
           'egg_info', 'easy_install', 'upload',
           '--single-version-externally-managed',
            )).intersection(sys.argv)) > 0:
    import setuptools
    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
    )
else:
    extra_setuptools_args = dict()

###############################################################################


class CleanCommand(Clean):
    description = "Remove build directories, and compiled file in the source tree"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('macuto'):
            for filename in filenames:
                if (filename.endswith('.so') or filename.endswith('.pyd')
                             or filename.endswith('.dll')
                             or filename.endswith('.pyc')):
                    os.unlink(os.path.join(dirpath, filename))


###############################################################################

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('macuto')

    return config


def setup_package():
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    version=VERSION,
                    download_url=DOWNLOAD_URL,
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 2',
                                 'Programming Language :: Python :: 2.6',
                                 'Programming Language :: Python :: 2.7',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.3',
                                 ],
                    cmdclass={'clean': CleanCommand},
                    **extra_setuptools_args)

    if (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1]
                 in ('--help-commands', 'egg_info', '--version', 'clean'))):

        # For these actions, NumPy is not required.
        #
        # They are required to succeed without Numpy for example when
        # pip is used to install Scikit when Numpy is not yet present in
        # the system.
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION
    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(name='macuto',
      description='MR Automatic Classification TOols',
      author='Alexandre Manhaes Savio',
      author_email='alexsavio@gmail.com',
      url='http://www.github.com/alexsavio/macuto/',
      **configuration(top_path='').todict())

    
