# coding=utf-8
#-------------------------------------------------------------------------------
#License GNU/GPL v3
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import numpy as np
import scipy.stats as stats


def calculate_stats (data):
    n_subjs = data.shape[0]

    feats  = np.zeros((n_subjs, 7))

    feats[:,0] = fs.max (axis=1)
    feats[:,1] = fs.min (axis=1)
    feats[:,2] = fs.mean(axis=1)
    feats[:,3] = fs.var (axis=1)
    feats[:,4] = np.median      (fs, axis=1)
    feats[:,5] = stats.kurtosis (fs, axis=1)
    feats[:,6] = stats.skew     (fs, axis=1)

    return feats

#-------------------------------------------------------------------------------
def calculate_hist3d (data, bins):
    """

    @param data:
    @param bins:
    @return:
    """
    n_subjs = data.shape[0]

    feats = np.zeros((n_subjs, bins*bins*bins))

    for s in np.arange(n_subjs):
        H, edges = np.histogramdd(data[s,], bins = (bins, bins, bins))
        feats[s,:] = H.flatten()

    return feats


def create_feature_sets (fsmethod, data, msk, y, outdir, outbasename, otype):
    """
    @param fsmethod:
    @param fsgrid:
    @param data:
    @param msk:
    @param y:
    @param outdir:
    @param outbasename:
    @param otype:
    @return:
    """
    np.savetxt (os.path.join(outdir, outbasename + '_labels.txt'), y, fmt="%.2f")

    outfname = os.path.join(outdir, outbasename)
    au.log.info('Creating ' + outfname)

    fs = data[:, msk > 0]

    if fsmethod == 'stats':
        feats = calculate_stats (fs)

    elif fsmethod == 'hist3d':
        feats = calculate_hist3d (fs)

    elif fsmethod == 'none':
        feats = fs

    #save file
    save_feats_file (feats, otype, outfname)
