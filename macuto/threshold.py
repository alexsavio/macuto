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


def binarise(data, lower_bound, upper_bound, inclusive=True):
    """

    @param data:
    @param lower_bound:
    @param upper_bound:
    @param inclusive:
    @return:
    """
    if inclusive:
        lowers = data >= lower_bound
        uppers = data <= upper_bound
    else:
        lowers = data >  lower_bound
        uppers = data <  upper_bound

    return lowers.astype(int) * uppers.astype(int)


def apply_threshold(values, thr, method='robust'):
    """
    @param values: numpy array

    @param thr: float
    between 0 and 100

    @param method: string
    Valid choices: 'robust', 'rank', 'percentile'

    @return: numpy array
    Thresholded array
    """
    if   method == 'robust':     return robust_range_threshold(values, thr)
    elif method == 'rank':       return rank_threshold        (values, thr)
    elif method == 'percentile': return percentile_threshold  (values, thr)


def find_histogram(vol, hist, mini, maxi, mask, use_mask):
    """
    For robust limits calculation

    @param vol: ndarray

    @param hist:
    @param mini:
    @param maxi:
    @param mask:
    @param use_mask:
    @return:
    hist,validsize
    """
    validsize = 0
    hist = np.zeros(hist.size, dtype=int)
    if mini == maxi:
        return -1

    fA = float(hist.size)/(maxi-mini)
    fB = (float(hist.size)*float(-mini)) / (maxi-mini)

    if use_mask:
        a = vol[mask > 0.5].flatten()
    else:
        a = vol.flatten()

    a = (a*fA + fB).astype(int)
    h = hist.size - 1

    for i in np.arange(a.size):
        hist[max(0, min(a[i], h))] += 1
        validsize += 1

    return hist, validsize


def is_symmetric(mat):
    """
    Returns true if mat is symmetric
    @param mat: numpy array
    @return:
    """
    return np.allclose(mat.T, mat)


def rank_threshold(distances, thr=95):
    """
    @param distances:
    @param thr:
    @return:
    """
    sorte     = distances.flatten().argsort()
    limit     = len(sorte) * thr/100
    distances[sorte[:limit-1]] = 0
    return distances


def percentile_threshold(distances, thr=95):
    """
    @param distances:
    @param thr:
    @return:
    """
    sels = np.select([distances >= np.percentile(distances, thr)], [distances])
    sels[np.isnan(sels)] = 0
    return sels


def find_thresholds(vol, mask, use_mask=True):
    """
    For robust limits calculation
    @param vol:
    @param mask:
    @param use_mask:
    @return: minval, maxval
    """
    hist_bins   = 1000
    hist        = np.zeros(hist_bins, dtype=int)
    max_jumps   = 10
    top_bin     = 0
    bottom_bin  = 0
    count       = 0
    jump        = 1
    lowest_bin  = 0
    highest_bin = hist_bins-1
    validsize   = 0

    thresh98 = float(0)
    thresh2  = float(0)
    mini     = float(0)
    maxi     = float(0)

    if use_mask:
        mini = vol[mask > 0].min()
        maxi = vol[mask > 0].max()
    else:
        mini = vol.min()
        maxi = vol.max()

    while jump == 1 or ((float(thresh98) - thresh2) < (maxi - mini)/10.):
        if jump > 1:
            bottom_bin = max(bottom_bin-1, 0)
            top_bin    = min(top_bin   +1, hist_bins-1)

            tmpmin = mini + (float(bottom_bin)/float(hist_bins)) * (maxi-mini)
            maxi   = mini + (float(top_bin+1) /float(hist_bins)) * (maxi-mini)
            mini   = tmpmin

        if jump == max_jumps or mini == maxi:
            if use_mask:
                mini = vol[mask > 0].min()
                maxi = vol[mask > 0].max()
            else:
                mini = vol.min()
                maxi = vol.max()

        hist, validsize = find_histogram(vol, hist, mini, maxi,
                                         mask, use_mask)

        if validsize < 1:
            thresh2  = mini
            minval   = mini
            thresh98 = maxi
            maxval   = maxi
            return minval, maxval

        if jump == max_jumps:
            validsize   -= np.round(hist[lowest_bin]) + np.round(hist[highest_bin])
            lowest_bin  += 1
            highest_bin -= 1

        if validsize < 0:
            thresh2  = mini
            thresh98 = mini

        fA = (maxi-mini)/float(hist_bins)

        count      = 0
        bottom_bin = lowest_bin
        while count < float(validsize)/50:
            count      += np.round(hist[bottom_bin])
            bottom_bin += 1
        bottom_bin -= 1
        thresh2     = mini + float(bottom_bin) * fA

        count   = 0
        top_bin = highest_bin
        while count < float(validsize)/50:
            count   += np.round(hist[top_bin])
            top_bin -= 1
        top_bin  += 1
        thresh98  = mini + (float(top_bin) + 1) * fA

        if jump == max_jumps:
            break

        jump += 1

    minval = thresh2
    maxval = thresh98
    return minval, maxval


def robust_min(vol, mask=''):
    """

    @param vol:
    @param mask:
    @return:
    """
    return find_thresholds(vol, mask)[0]


def robust_max (vol, mask=''):
    """

    @param vol:
    @param mask:
    @return:
    """
    return find_thresholds(vol, mask)[1]


def threshold (data, lower_bound, upper_bound, inclusive=True):
    """
    @param data:
    @param lower_bound:
    @param upper_bound:
    @param inclusive:
    @return:
    """
    mask = binarise(data, lower_bound, upper_bound, inclusive)
    return data * mask


def robust_range_threshold(vol, thrP):
    """

    @param vol:
    @param thrP:
     thrP should go within [0, 100]
    @return:
    """
    mask       = binarise(vol, 0, vol.max()+1, False)
    limits     = find_thresholds(vol, mask)
    lowerlimit = limits[0] + float(thrP)/100*(limits[1]-limits[0])
    out        = threshold(vol, lowerlimit, vol.max()+1, True)
    return out #out.astype(vol.dtype)
