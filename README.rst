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

GIT
~~~

You can check the latest sources with the command::

    git clone https://github.com/alexsavio/macuto.git

or if you have write privileges::

    git clone git@github.com:alexsavio/macuto.git


Testing
-------

TODO
