import numpy as np
from math import log
from scipy import histogram, digitize
from collections import defaultdict

#---------------------------------------------------------------------
import math
import numpy
import scipy
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad

# Constants
MIN_DOUBLE = 4.9406564584124654e-324
                    # The minimum size of a Float64; used here to prevent the
                    #  logarithmic function from hitting its undefined region
                    #  at its asymptote of 0.
INF=1E12  # The floating-point representation for "infinity"

# x and y are previously defined as collections of
# floating point values with the same length

# Kernel estimation
gkde_x = gaussian_kde(x)
gkde_y = gaussian_kde(y)

if len(binned_x) != len(binned_y) and len(binned_x) != len(x):
    x.append(x[0])
    y.append(y[0])

gkde_xy = gaussian_kde([x,y])
mutual_info = lambda a,b: gkde_xy([a,b]) * \
    math.log((gkde_xy([a,b]) / ((gkde_x(a) * gkde_y(b)) + MIN_DOUBLE)) \
        + MIN_DOUBLE)

# Compute MI(X,Y)
(minfo_xy, err_xy) = \
    dblquad(mutual_info, -INF, INF, lambda a: 0, lambda a: INF)

print('minfo_xy = ', minfo_xy)

#---------------------------------------------------------------------
from sklearn.metrics import mutual_info_score


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


#----------------------------------------------------------------------
log2 = lambda x: log(x, 2)

def mutual_information(x, y):
    return entropy(y) - conditional_entropy(x,y)


def conditional_entropy(x, y):
    """
    x: vector de numeros reales
    y: vector de numeros enteros

    devuelve H(Y|X)
    """
    # discretizacion de X 
    hx, bx = histogram(x, bins=x.size/10, density=True)

    Py = compute_distribution(y)
    Px = compute_distribution(digitize(x, bx))

    res = 0
    for ey in set(y):
        # P(X | Y)
        x1 = x[y == ey]
        condPxy = compute_distribution(digitize(x1,  bx))

        for k, v in condPxy.iteritems():
            res += (v * Py[ey] * (log2(Px[k]) - log2(v * Py[ey])))
    return res


def entropy(y):
    """
    Computa la entropia de un vector discreto
    """
    # P(Y)
    Py = compute_distribution(y)
    res = 0.0
    for k, v in Py.iteritems():
        res += v*log2(v)
    return -res


def compute_distribution(v):
    """
    v: vector de valores enteros

    devuelve un diccionario con la probabilidad de cada valor
    computado como la frecuencia de ocurrencia
    """
    d = defaultdict(int)
    for e in v:
        d[e] += 1
    s = float(sum(d.values()))
    return dict((k, v/s) for k, v in d.items())
