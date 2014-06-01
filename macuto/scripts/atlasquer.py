#!/usr/bin/env python

import os
import sys
import argparse
import logging

import nibabel as nib

from macuto.atlas.atlas_group import AtlasGroup
from macuto.scriptutils import whoami

#logging config
logging.basicConfig(level=logging.DEBUG, filename='atlasquerpy.log',
                    format="%(asctime)-15s %(message)s")
log = logging.getLogger(__name__)

'''
cd ~/Dropbox/Documents/work/atlasquerpy
python atlasquerpy -m /home/alexandre/Data/cobre/presels.nii.gz -a "MNI Structural Atlas"
python atlasquerpy -m /home/alexandre/Dropbox/Documents/locals.nii.gz -a "MNI Structural Atlas"
'''


#from IPython.core.debugger import Tracer; debug_here = Tracer()
#-------------------------------------------------------------------------------
def set_parser():
    parser = argparse.ArgumentParser(description='Atlasquerpy')
    parser.add_argument('-a', '--atlas', dest='atlas', required=True,
                        help='name of atlas to use')
    parser.add_argument('-V', '--verbose', dest='verbose', required=False,
                        action='store_true', default=False,
                        help='switch on diagnostic messages')
    parser.add_argument('-m', '--mask', dest='mask', required=False, 
                        default = '',
                        help='a mask image to use during structural lookups')
    parser.add_argument('-b', '--binarise', dest='binarise', required=False, 
                        action='store_true', default = False,
                        help='switch on binarization of the mask.')
    parser.add_argument('-t', '--type', dest='type', required=False, 
                        choices=['avgprob', 'roiover'], default='avgprob',
                        help='''Type of measure: average mask or coordinate 
                              Probability; or ROI overlapping percentage''')
    parser.add_argument('-c', '--coords', dest='coords', required=False, 
                        help='''specify coordinates of the point of interest 
                             (as mm coordinates): <X>,<Y>,<Z>''')
    parser.add_argument('-p', '--precision', dest='precision', required=False, 
                        default=4,
                        help='''specify the precision of the floats that will 
                             be printed when using -m''')
    parser.add_argument('--dumpatlases', dest='dumpatlases', required=False, 
                        action='store_true', default=False,
                        help='Dump a list of the available atlases')

    return parser
#-------------------------------------------------------------------------------


def main(argv=None):

    parser  = set_parser()

    try:
       args = parser.parse_args()
    except argparse.ArgumentError as exc:
       log.error(exc.message)
       parser.error(str(exc.message))
       return -1

    log.info('atlasquer.py {0} {1} {2}'.format(os.path.basename(__file__),
                                               whoami(),
                                               locals()))

    dumpatlases = args.dumpatlases
    precision = args.precision
    atlas_name = args.atlas
    mask_file = args.mask
    binarise = args.binarise
    verbose = args.verbose
    coords = args.coords
    qtype = args.type

    atlas_group = AtlasGroup()

    if dumpatlases:
        print(atlas_group.atlases.keys())
        return 0

    if verbose:
        log.info('Using atlas: ' + atlas_name)

    atlas = atlas_group.get_atlas_by_name(atlas_name)
    if atlas is None:
        log.error('Invalid atlas name. Try one of:')
        print(atlas_group.atlases.keys())
        return 1

    if mask_file != '':
        try:
            mask_img = nib.load(mask_file)
        except:
            log.error('Problem reading mask file ' + mask_file)
            return 1

        if verbose:
            log.info('Working with mask ' + mask_file)

        if binarise:
            mask_vol = mask_img.get_data()
            mask_vol[mask_vol > 0] = 1

        lidx = atlas.get_labels_ids()
        for li in lidx:
            value = 0.0
            if qtype == 'avgprob':
                value = atlas.get_avg_probability(mask_img, li)
            elif qtype == 'roiover':
                value = atlas.get_roi_overlap(mask_img, li)

            struct_name = atlas.get_structure_name(li)

            if verbose:
                log.debug(str(li))

            if value > 0:
                val_text = "%.*f" % (precision, round(value, precision))
                log.info(struct_name + ':' + val_text)

    elif coords != '':
        try:
            k = coords.split(',')
            x, y, z = float(k[0]), float(k[1]), float(k[2])
        except:
            log.error('Problem parsing given coordinates, try <x>,<y>,<z>')
            return 1

        if verbose:
            log.debug('Working with coords: ' + str(x) + ',' + str(y) + ',' + str(z))

        try:
            print(atlas.get_description(x, y, z))
        except Exception as exc:
            log.error(exc)
            log.error(sys.exc_info())
            return 1

    return 0

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())


