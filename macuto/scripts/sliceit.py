#!/usr/bin/env python

import os
import os.path as op
import logging

import baker

from macuto.files.search import recursive_find_match
from macuto.scriptutils import whoami
from macuto.plotting.render import (slicesdir_oneset,
                                    slicesdir_paired_overlays,
                                    create_imglist_html)


#logging config
logging.basicConfig(level=logging.DEBUG, filename='sliceit.log',
                    format="%(asctime)-15s %(message)s")
log = logging.getLogger('sliceit')


@baker.command(default=True,
               shortopts={'regex1': '1',
                          'regex2': '2',
                          'max_jumps': 'm',
                          'red_outline': 'r',
                          'axis': 'a'})
def vols(inputdir='', outdir='', regex1='', regex2='',
         red_outline=False, max_jumps=3, axis=1, dpi=150):
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

    :param red_outline: boolean, optional
    Set this flag to plot only an outline of the files in regex2.

    :param max_jumps: int, optional
    Number of folder jumps upwards to recursively look for
    regex2 file matches.

    :param axis: int
    Cut-axis. Choices: '0' for x-axis, '1' for y-axis, '2' for z-axis

    :param dpi: int
    Dots-per-inch of the result images.
    """
    #assert(op.exists(inputdir))
    log.info('Running {0} {1} {2}'.format(op.basename(__file__), whoami(),
                                          locals()))

    assert(op.isdir(inputdir))

    #look for regex1 matches
    base_files = recursive_find_match(inputdir, regex1)
    base_files.sort()

    if len(base_files) == 0:
        log.error('Could not find files that matched {0} within folder {1}'.format(regex1, inputdir))
        return -1

    #check if output folder exists
    outdir = op.abspath(op.realpath(outdir))
    if not op.exists(outdir):
        log.info('Creating folder {0}'.format(outdir))
        os.makedirs(outdir)

    if len(regex2) == 0:
        #create slices for regex1 matches only
        out_images = slicesdir_oneset(outdir, base_files, dpi=dpi, 
                                      volaxis=axis)
    else:
        #look for matches to regex2 and create slices for both
        overlay_files = [find_nearest_match(bf, regex2, max_jumps) for bf in base_files]
        out_images = slicesdir_paired_overlays(outdir, base_files, 
                                               overlay_files,
                                               is_red_outline=red_outline,
                                               dpi=dpi)

    create_imglist_html(outdir, out_images)

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
    if not op.isabs(reffile):
        reffile = op.abspath(reffile)

    basedir = op.abspath(op.realpath(op.dirname(reffile)))
    for i in range(max_jumps):
        matches = recursive_find_match(basedir, pattern)
        if len(matches) > 0:
            return matches[0]
        basedir = basedir.parent

    return ''

if __name__ == '__main__':
    baker.run()
