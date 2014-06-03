# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import os
import numpy as np
import scipy.stats as stats
import logging

from ..threshold import apply_threshold
from ..storage import ExportData

log = logging.getLogger(__name__)


def calculate_stats(data):
    """

    @param data: numpy array
    Shape: n_samples x n_features

    @return:
    """
    n_subjs = data.shape[0]

    feats  = np.zeros((n_subjs, 7))

    feats[:, 0] = data.max (axis=1)
    feats[:, 1] = data.min (axis=1)
    feats[:, 2] = data.mean(axis=1)
    feats[:, 3] = data.var (axis=1)
    feats[:, 4] = np.median      (data, axis=1)
    feats[:, 5] = stats.kurtosis (data, axis=1)
    feats[:, 6] = stats.skew     (data, axis=1)

    return feats


def calculate_hist3d(data, bins):
    """

    @param data: numpy array
    Shape: n_samples x n_features

    @param bins:
    @return:
    """
    n_subjs = data.shape[0]

    feats = np.zeros((n_subjs, bins*bins*bins))

    for s in np.arange(n_subjs):
        h, edges = np.histogramdd(data[s, ], bins=(bins, bins, bins))
        feats[s, :] = h.flatten()

    return feats


def create_feature_sets(fsmethod, data, msk, y, outdir, outbasename, otype='.h5'):
    """
    @param fsmethod:

    @param fsgrid:

    @param data: numpy array
    Shape: n_samples x n_features

    @param msk:

    @param y:

    @param outdir:

    @param outbasename:

    @param otype: string
    Valid values '.pyshelf', '.mat', '.hdf5' or '.h5'

    @return:
    """
    np.savetxt(os.path.join(outdir, outbasename + '_labels.txt'), y, fmt="%.2f")

    outfname = os.path.join(outdir, outbasename)
    log.info('Creating ' + outfname)

    fs = data[:, msk > 0]

    if fsmethod == 'stats':
        feats = calculate_stats(fs)

    elif fsmethod == 'hist3d':
        feats = calculate_hist3d(fs)

    elif fsmethod == 'none':
        feats = fs

    #save file
    ExportData.save_variables(outfname + otype, {'feats': feats})


def pearson_correlation(x, y):
    """
    Calculates for each feature in X the
    pearson correlation with y.

    @param x: numpy array
    Shape: n_samples x n_features

    @param y: numpy array or list
    Size: n_samples

    @return: numpy array
    Size: n_features
    """
    return distance_computation(x, y, stats.pearsonr)


#-------------------------------------------------------------------------------
def distance_computation(x, y, dist_function):
    """
    Calculates for each feature in X the
    given dist_function with y.

    @param x: numpy array
    Shape: n_samples x n_features

    @param y: numpy array or list
    Size: n_samples

    @param dist_function: function
    distance function

    @return: numpy array
    Size: n_features

    @note:
    Apply any given 1-D distance function to X and y.
    Have a look at:
    http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    """
    #number of features
    n_feats = x.shape[1]

    #creating output volume file
    p = np.zeros(n_feats)

    #calculating dist_function across all subjects
    for i in list(range(x.shape[1])):
        p[i] = dist_function(x[:, i], y)[0]

    p[np.isnan(p)] = 0

    return p


def bhattacharyya_dist(x, y):
    """
    Univariate Gaussian Bhattacharyya distance
    between the groups in X, labeled by y.

    @param x: numpy array
    Shape: n_samples x n_features

    @param y: numpy array or list
    Size: n_samples

    @return:
    """
    classes = np.unique(y)
    n_class = len(classes)
    n_feats = x.shape[1]

    b = np.zeros(n_feats)
    for i in np.arange(n_class):
        for j in np.arange(i + 1, n_class):
            if j > i:
                xi = x[y == i, :]
                xj = x[y == j, :]

                mi = np.mean (xi, axis=0)
                mj = np.mean (xj, axis=0)

                vi = np.var  (xi, axis=0)
                vj = np.var  (xj, axis=0)

                si = np.std  (xi, axis=0)
                sj = np.std  (xj, axis=0)

                d = 0.25 * (np.square(mi - mj) / (vi + vj)) + 0.5 * (np.log((vi + vj) / (2*si*sj)))
                d[np.isnan(d)] = 0
                d[np.isinf(d)] = 0

                b = np.maximum(b, d)

    return b


def welch_ttest(x, y):
    """
    Welch's t-test between the groups in X, labeled by y.

    @param x: numpy array
    Shape: n_samples x n_features

    @param y: numpy array or list
    Size: n_samples

    @return:
    """
    classes = np.unique(y)
    n_class = len(classes)
    n_feats = x.shape[1]

    b = np.zeros(n_feats)
    for i in np.arange(n_class):
        for j in np.arange(i+1, n_class):
            if j > i:
                xi = x[y == i, :]
                xj = x[y == j, :]
                yi = y[y == i]
                yj = y[y == j]

                mi = np.mean (xi, axis=0)
                mj = np.mean (xj, axis=0)

                vi = np.var  (xi, axis=0)
                vj = np.var  (xj, axis=0)

                n_subjsi = len(yi)
                n_subjsj = len(yj)

                t = (mi - mj) / np.sqrt((np.square(vi) / n_subjsi) + (np.square(vj) / n_subjsj))
                t[np.isnan(t)] = 0
                t[np.isinf(t)] = 0

                b = np.maximum(b, t)

    return b


def feature_selection(x, y, method, thr=95, dist_function=None,
                      thr_method='robust'):
    """
    Parameters
    ----------
    @param: x             : data ([n_samps x n_feats] matrix)
    @param: y             : class labels
    @param: method        : distance measure: 'pearson', 'bhattacharyya', 'welcht', ''
                            if method == '', will use dist_function
    @param: thr           : percentile distance threshold
    @param: dist_function :
    @param: thr_method    : method for thresholding: 'none', 'robust', 'ranking', 'percentile'
                            See .threshold.apply_threshold docstring.

    Returns
    -------
    @return m          : distance measure (thresholded or not)
    """
    #pre feature selection, measuring distances
    #Pearson correlation
    if method == 'pearson':
        log.info('Calculating Pearson correlation')
        m = np.abs(pearson_correlation(x, y))

    #Bhattacharyya distance
    elif method == 'bhattacharyya':
        log.info('Calculating Bhattacharyya distance')
        m = bhattacharyya_dist (x, y)

    #Welch's t-test
    elif method == 'welcht':
        log.info("Calculating Welch's t-test")
        m = welch_ttest(x, y)

    elif method == '':
        log.info('Calculating distance between data and class labels')
        #http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        m = distance_computation(x, y, dist_function)

    #if all distance values are 0
    if not m.any():
        log.info("No differences between groups have been found. "
                 "Are you sure you want to continue?")
        return m

    #threshold data
    if thr_method != 'none':
        return apply_threshold(m, thr, thr_method)

    return m