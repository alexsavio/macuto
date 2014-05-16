.. -*- mode: rst -*-

macuto
======

.. image:: https://secure.travis-ci.org/alexsavio/macuto.png?branch=master
    :target: https://travis-ci.org/alexsavio/macuto
.. image:: https://coveralls.io/repos/alexsavio/macuto/badge.png
    :target: https://coveralls.io/r/alexsavio/macuto

Series of tools to manage MRI data, extract features from them and easily perform supervised classification with cross-validation.
It uses nibabel, nitime, scikit-learn and h5py.

Dependencies
============

Please see the requirements.txt file.

Install
=======

Before installing it, you need all the requirements installed.
These are listed in the requirements.txt files.
The best way to install them is running the following command:

    <pre><code>
    for r in `cat macuto/requirements.txt`; do pip install $r; done
    </code></pre>

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install


Development
===========

Code
----

Gitlab
~~~~~~

You can check the latest sources with the command::

    git clone https://158.227.114.158/alexandre/macuto.git

or if you have write privileges::

    git clone git@158.227.114.158:alexandre/macuto.git

If you are going to create patches for this project, create a branch for it 
from the develop branch.

The master branch is exclusive for stable releases.


Github
~~~~~~

An older version of the project is also available in Github.

You can check the latest sources with the command::

    git clone https://github.com/alexsavio/macuto.git

or if you have write privileges::

    git clone git@github.com:alexsavio/macuto.git


Testing
-------

TODO
