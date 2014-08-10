# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#Authors:
# Alexandre Manhaes Savio <alexsavio@gmail.com>
# Darya Chyzhyk <darya.chyzhyk@gmail.com>
# Borja Ayerdi <ayerdi.borja@gmail.com>
# Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
# Neurita S.L.
#
# BSD 3-Clause License
#
# 2014, Alexandre Manhaes Savio
# Use this at your own risk!
#------------------------------------------------------------------------------

import numpy as np
import scipy.spatial.distance as scipy_dist

from ..utils import Printable

class DistanceMeasure(Printable): pass
    #def fit_transform(self):
    #    raise NotImplementedError


class ScipyDistance(DistanceMeasure):
    """
    This is a wrapper class for scipy distance measures.
    """

    def __init__(self, dist_function):
        self._set_dist_function(dist_function)

    def _set_dist_function(self, dist_function):
        """
        :param dist_function: function
        distance function

        :return:
        """
        if not hasattr(dist_function, '__call__'):
            raise ValueError('dist_function must be a function')

        if not hasattr(scipy_dist, dist_function.__name__):
            raise ValueError('dist_function must be a scipy.spatial.distance '
                             'function.')

        self._dist_function = dist_function

    def fit_transform(self, x, y):
        """
        Apply any given 1-D distance function to x and y.
        Have a look at:
        http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

        :param x: numpy array
        Shape: n_samples x n_features

        :param y: numpy array or list
        Size: n_samples

        :return: numpy array
        Size: n_features
        """
        return distance_computation(x, y, self._dist_function)


class PearsonCorrelationDistance(ScipyDistance):
    """
    The absolute Pearson's correlation between each feature in X and the
    class labels in y.
    """

    def __init__(self):
        ScipyDistance.__init__(self, scipy_dist.pearsonr)

    def fit_transform(self, x, y):
        """
        Calculates for each feature in X the
        pearson correlation with y.

        :param x: numpy array
        Shape: n_samples x n_features

        :param y: numpy array or list
        Size: n_samples

        :return: numpy array
        Size: n_features
        """
        return np.abs(ScipyDistance.fit_transform(self, x, y))


class WelchTestDistance(DistanceMeasure):
    """
    Welch's t-test between the groups in X, labeled by y.
    """

    def __init__(self):
        pass

    def fit_transform(self, x, y):
        """
        Welch's t-test between the groups in X, labeled by y.

        :param x: numpy array
        Shape: n_samples x n_features

        :param y: numpy array or list
        Size: n_samples

        :return:numpy array
        Size: n_features
        """

        return welch_ttest(x, y)


class BhatacharyyaGaussianDistance(DistanceMeasure):
    """
    Univariate Gaussian Bhattacharyya distance
    between the groups in X, labeled by y.
    """

    def __init__(self):
        pass

    def fit_transform(self, x, y):
        """
        Univariate Gaussian Bhattacharyya distance
        between the groups in X, labeled by y.

        :param x: numpy array
        Shape: n_samples x n_features

        :param y: numpy array or list
        Size: n_samples

        :return: numpy array
        Size: n_features
        """
        return bhattacharyya_dist(x, y)


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
    return distance_computation(x, y, scipy_dist.pearsonr)


#------------------------------------------------------------------------------
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

                d = 0.25 * (np.square(mi - mj) / (vi + vj)) + \
                    0.5 * (np.log((vi + vj) / (2*si*sj)))
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

                t = (mi - mj) / np.sqrt((np.square(vi) / n_subjsi) +
                                        (np.square(vj) / n_subjsj))
                t[np.isnan(t)] = 0
                t[np.isinf(t)] = 0

                b = np.maximum(b, t)

    return b

