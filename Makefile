# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-pyc:
	find macuto -name "*.pyc" | xargs rm -f

clean-so:
	find macuto -name "*.so" | xargs rm -f
	find macuto -name "*.pyd" | xargs rm -f
	find macuto -name "__pycache__" | xargs rm -rf

clean-build:
	rm -rf build
	rm -rf dist

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(NOSETESTS) -s -v macuto
test-doc:
	$(NOSETESTS) -s -v doc/


trailing-spaces:
	find macuto -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

doc: inplace
	make -C doc html

doc-noplot: inplace
	make -C doc html-noplot

code-analysis:
	flake8 macuto | grep -v __init__ | grep -v external
	pylint -E -i y macuto/ -d E1103,E0611,E1101
