#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

import sys
import argparse
import numpy as np
import nibabel as nib
import scipy.ndimage as scn

#-------------------------------------------------------------------------------
def set_parser():
    parser = argparse.ArgumentParser(description='Empties each ROI in the 3D input volume and saves the result in the output volume.')
    parser.add_argument('-i', '--in', dest='input', required=True, help='input file')
    parser.add_argument('-o', '--out', dest='output', required=True, help='output file')
    return parser

#-------------------------------------------------------------------------------
def add_extension_if_needed (fpath, ext, check_if_exists=False):

   if fpath.find(ext) < 0:
     fpath += ext

   if check_if_exists:
      if not os.path.exists (fpath):
         err = 'File not found: ' + fpath
         raise IOError(err)

   return fpath

#-------------------------------------------------------------------------------
def mayavi_visualize_vol (vol):
    from mayavi import mlab

    mlab.clf()
    s = mlab.pipeline.scalar_field(vol)
    ipw_x = mlab.pipeline.image_plane_widget(s, plane_orientation='x_axes')
    ipw_y = mlab.pipeline.image_plane_widget(s, plane_orientation='y_axes')

#-------------------------------------------------------------------------------
def main(argv=None):

    parser  = set_parser()

    try:
        args = parser.parse_args ()
    except argparse.ArgumentError as exc:
        print(exc.message + '\n' + exc.argument)
        parser.error(str(exc.message))
        return 0

    ifname  = args.input.strip()
    ofname   = args.output.strip()

    ofname = add_extension_if_needed(ofname, '.nii.gz')

    aff = nib.load(ifname).get_affine()
    vol = nib.load(ifname).get_data()
    out = np.zeros(vol.shape, dtype=vol.dtype)

    if vol.ndim == 2:
        kernel = np.ones([3,3], dtype=int)
    elif vol.ndim == 3:
        kernel = np.ones([3,3,3], dtype=int)
    elif vol.ndim == 4:
        kernel = np.ones([3,3,3,3], dtype=int)

    vals = np.unique(vol)
    vals = vals[vals != 0]

    for i in vals:
        roi  = vol == i
        hits = scn.binary_hit_or_miss (roi, kernel)
        roi[hits] = 0
        out[roi > 0] = i

    ni = nib.Nifti1Image(out, aff)
    nib.save(ni, ofname)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
