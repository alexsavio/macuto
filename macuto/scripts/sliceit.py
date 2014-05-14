#!/usr/bin/env python

import os
import baker
import logging
from path import path
from macuto.files.search import recursive_find_match
from macuto.scriptutils import whoami

#logging config
logging.basicConfig(level=logging.DEBUG, filename='sliceit.log',
                    format="%(asctime)-15s %(message)s")
log = logging.getLogger('sliceit')


@baker.command(default=True,
               shortopts={'regex1': '1',
                          'regex2': '2',
                          'max_jumps': 'm'})
def vols(inputdir, outdir, regex1, regex2='', max_jumps=3, dpi=150):
    """
    Creates a folder named outdir with a html file and png images of slices
    of each of nifti file that matches regex1.
    The regex2 files are overlay images.

    If it is the path to only one file, this file will be used on every
    file that matches regex1.

    If regex2 is a regular expression, this will look for the nearest file
    to the corresponding regex1 that matches regex2..

    :param inputdir: string
    Path to where start looking for the regular expressions.

    :param outdir: string
    Path to folder where to put the html and image files.
    If it doesn't exist, it will be created.

    :param regex1: string
    Regular expression for the base volumes.

    :param regex2: string
    Regular expression to for the overlay volumes.
    Can also be the path to one file.

    :param max_jumps: int
    Number of folder jumps upwards to recursively look for
    regex2 file matches.

    :param dpi: int
    Dots-per-inch of the result images.
    """
    #assert(os.path.exists(inputdir))
    log.info('Running {0} {1} {2}'.format(os.path.basename(__file__),
                                    whoami(),
                                    locals()))

    from macuto.render import (slicesdir_oneset,
                               slicesdir_paired_overlays)
    from macuto.files.search import (recursive_find_match)

    base_files = recursive_find_match(inputdir, regex1)

    if len(base_files) == 0:
        log.error('Could not find files that matched {0} in {1}'.format(regex1, inputdir))

    outdir = path(outdir)
    if not os.path.exists(outdir):
        outdir.makedirs_p()

    if len(regex2) == 0:
        slicesdir_oneset(outdir, base_files, dpi=dpi)
    else:
        overlay_files = [find_nearest_match(bf, regex2, max_jumps) for bf in base_files]
        slicesdir_paired_overlays(outdir, base_files, overlay_files, dpi=dpi)

    return 0


def find_nearest_match(basefile, pattern, max_jumps=3):
    """
    Return the first that matches the pattern near to basefile.
    Jump max_jumps upwards the folders to recursively look for pattern matches.
    Will return empty if none is found.

    :param basefile: string
    Path to the reference file

    :param pattern: string
    Regular expression to look for

    :param max_jumps: int
    Maximum number of upwards jumps to recursively look for the file.

    :return: string
    Path to the first found match.
    """
    reffile = basefile
    if not os.path.isabs(reffile):
        reffile = os.path.abspath(reffile)

    basedir = path(os.path.dirname(reffile))
    for i in range(max_jumps):
        matches = recursive_find_match(basedir, pattern)
        if len(matches) > 0:
            return matches[0]
        basedir = basedir.parent

    return ''

if __name__ == '__main__':
    baker.run()