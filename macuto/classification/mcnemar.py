import logging

import numpy as np
from scipy import stats
from math import sqrt

from ..more_collections import ItemSet


log = logging.getLogger(__name__)


# class ClassificationPredictionsSet(object):
#
#     def __init__(self, targets, predictions):
#         """
#         """
#         self.targets = np.array()
#         self.predictions = dict(('svm', np.array()))

#prep_mcnemar_alex
def prep_mcnemar_alex(y_vals, c1_preds, c2_preds):

    assert(len(y_vals) == len(c1_preds) == len(c2_preds))

    a, b, c, d = 0, 0, 0, 0

    for idx in range(len(y_vals)):
        y = y_vals[idx]
        c1 = c1_preds[idx]
        c2 = c2_preds[idx]
        if c1 == c2:
            if c1 == y:
                a +=1
            else:
                d +=1
        else:
            if c1 == y:
                b += 1
            else:
                c += 1

    return a, b, c, d

#prep_mcnemar_borja
def entrenar_clasificador():
    # Entrenar clasificador
    from sklearn import tree
    out = np.array([1,1,2,2,3,3,4,4])
    inp = np.matrix('1 1 1; 0 0 0; 1 0 0;0 1 1;0 0 1;1 1 0;0 1 0;1 0 1')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(inp, out)
    return clf


def prep_mcnemar_borja(clf, y_vals, c1_preds, c2_preds):
    res = clf.predict(np.array([y_vals, c1_preds, c2_preds]).T)
    return sum(res==1), sum(res==3), sum(res==4), sum(res==2)


def mcnemar(A, B, C, D, alpha=0.05, onetailed=False, verbose=False):
    """
    Performs a mcnemar test.
       A,B,C,D- counts in the form
        A    B  A+B
        C    D  C+D
       A+C  B+D  n
 
       alpha - level of significance
       onetailed -False for two-tailed test
                  True for one-tailed test 
    Returns True if Null hypotheses pi1 == pi2 is accepted
    else False.
    """
    tot = float(A + B + C + D)
    try:
        Z = (B-C)/ sqrt(B+C)
    except ZeroDivisionError as zd:
        log.error('Division by zero')
 
    if verbose:
        print "McNemar Test with A,B,C,D = ", A,B, C,D
        print "Ratios:p1, p2 = ",(A+B)/tot, (C + D) /tot
        print "Z test statistic Z = ", Z
 
 
    if onetailed:
       if (B-C> 0):
         zcrit2 = stats.norm.ppf(1-alpha)
         result = True if (Z < zcrit2)else False
         if verbose:
            print "Upper critical value=", zcrit2
            print "Decision:",  "Accept " if (result) else "Reject ",
            print "Null hypothesis at alpha = ", alpha
       else:
         zcrit1 = stats.norm.ppf(alpha)
         result = False if (Z < zcrit1) else False
         if verbose:
            print "Lower critical value=", zcrit1
            print "Decision:",  "Accept " if (result) else "Reject ",
            print "Null hypothesis at alpha = ", alpha
 
 
    else:
       zcrit1 = stats.norm.ppf(alpha/2.0)
       zcrit2 = stats.norm.ppf(1-alpha/2.0)
 
       result = True if (zcrit1 < Z < zcrit2) else False
       if verbose:
          print "Lower and upper critical limits:", zcrit1, zcrit2
          print "Decision:","Accept " if result else "Reject ",
          print "Null hypothesis at alpha = ", alpha
 
    return result
 
print mcnemar(200, 50, 100, 650,alpha = 0.05, verbose=True)

def mcnemar2(a,b,c,d):
    """
    Input args:
       a, b, c, d- frequencies
    Output:
       pvalue of test.
    """
    chi2testval = (abs(a-d) - 1)**2/ (a + d)
    df = 1
    pvalue = 1 - stats.chi2.cdf(chi2testval, df)
    return pvalue



if __name__ == '__main__':

    clf = entrenar_clasificador()

    y = np.array([1,0,1,0,1,0,1,0])
    c1 = np.array([1,0,0,1,1,0,0,1])
    c2 = np.array([1,0,0,1,0,1,1,0])

    a, b, c, d = prep_mcnemar_alex(y, c1, c2)
    a, b, c, d = prep_mcnemar_borja(clf, y, c1, c2)

    n = 100000
    y = np.array([1]*(n/2) + [0]*(n/2))
    c1 = np.array([1]*(n/4) + [0]*(n/4) + [1]*(n/4) + [0]*(n/4))
    c2 = np.array([0]*(n/4) + [1]*(n/4) + [0]*(n/4) + [1]*(n/4))
    
    a, b, c, d = prep_mcnemar_alex(y, c1, c2)
    a, b, c, d = prep_mcnemar_borja(clf, y, c1, c2)
    
    c2 = np.array([0]*(n/8) + [1]*(n/8) + [0]*(n/8) + [1]*(n/8) + [0]*(n/8) + [1]*(n/8) + [0]*(n/8) + [1]*(n/8))
