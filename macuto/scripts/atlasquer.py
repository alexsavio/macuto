#!/usr/bin/python

import sys
import argparse

import nibabel as nib
from atlas_group import AtlasGroup

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
       args = parser.parse_args ()
    except argparse.ArgumentError, exc:
       print(exc.message + '\n' + exc.argument)
       parser.error(str(exc.message))
       return -1

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
        print('Using atlas: ' + atlas_name)

    atlas = atlas_group.get_atlas_by_name(atlas_name)
    if atlas is None:
        print('Invalid atlas name. Try one of:')
        print(atlas_group.atlases.keys())

    if mask_file != '':
        try:
            mask_img = nib.load(mask_file)
        except:
            print('Problem reading mask file ' + mask_file)
            return 1

        if verbose:
            print('Working with mask ' + mask_file)

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
                print(str(li))

            if value > 0:
                val_text = "%.*f" % (precision, round(value, precision))
                print(struct_name + ':' + val_text)

    elif coords != '':
        try:
            k = coords.split(',')
            x, y, z = float(k[0]), float(k[1]), float(k[2])
        except:
            print('Problem parsing given coordinates, try <x>,<y>,<z>')

        if verbose:
            print('Working with coords: ' + str(x) + ',' + str(y) + ',' + str(z))

        try:
            print(atlas.get_description(x, y, z))
        except:
            print('Unknown exception.')
            return 1

    return 0

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())


