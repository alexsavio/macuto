
import numpy as np
from math import sqrt
from scipy import stats


class ClassifiersMcNemarTest(object):
    """Class helper to calculate McNemar's significance test between two 
    classification results.
    """

    def __init__(self):
        self._a, self._b, self._c, self._d = None, None, None, None

    def fit_transform(self, targets, c1_predictions, c2_predictions, alpha=0.1,
                      onetailed=True):
        """
        Parameters
        ----------
        targets : np.ndarray or list
            Classification target values

        c1_predictions : np.ndarray or list
            First classifier prediction values

        c1_predictions: np.ndarray or list
            Second classifier prediction values

        alpha : float
        Level of significance
        
        onetailed: bool
                    False for two-tailed test
                    True for one-tailed test 

        Returns
        -------
        McNemar's significance test value

        """
        self.fit(targets, c1_predictions, c2_predictions)
        return self.transform(alpha=alpha, onetailed=onetailed)

    def fit(self, targets, c1_predictions, c2_predictions):
        """
        Parameters
        ----------
        targets : np.ndarray or list
            Classification target values

        c1_predictions : np.ndarray or list
            First classifier prediction values

        c1_predictions: np.ndarray or list
            Second classifier prediction values
        """
        self._a, self._b, self._c, self._d = get_mcnemar_abcd(targets,
                                                              c1_predictions,
                                                              c2_predictions)

    def transform(self, alpha=0.1, onetailed=True):
        """Returns the boolean value of the McNemar's Test of c1 vs. c2.

        Parameters
        ----------
        alpha : float
            Level of significance

        onetailed: bool
            False for two-tailed test
            True for one-tailed test

        Returns
        -------
        bool
            True if Null hypotheses pi1 == pi2 is accepted
            else False.
        """
        return mcnemar(self._a, self._b, self._c, self._d,
                       alpha=alpha, onetailed=onetailed)

    def is_significative(self, targets, c1_predictions, c2_predictions,
                         alpha=0.1, onetailed=True):
        """
        See fit_transform docstring

        Parameters
        ----------
        targets:

        c1_predictions:

        c2_predictions:

        alpha:

        onetailed:

        Returns
        -------
        """
        return not self.fit_transform(targets, c1_predictions, c2_predictions,
                                      alpha=alpha, onetailed=onetailed)


def get_mcnemar_abcd(y_vals, c1_preds, c2_preds):
    """Returns the McNemar's confusion matrix values A, B, C, D

    Parameters
    ----------
    y_vals : np.ndarray or list
        Classification target values

    c1_preds : np.ndarray or list
        First classifier predictions

    c2_preds: np.ndarray or list
        Second classifier values

    Returns
    -------
    Integers: a, b, c, d

    """
    assert(len(y_vals) == len(c1_preds) == len(c2_preds))

    a, b, c, d = 0, 0, 0, 0

    for idx in range(len(y_vals)):
        y = y_vals[idx]
        c1 = c1_preds[idx]
        c2 = c2_preds[idx]
        if c1 == c2:
            if c1 == y:
                a += 1
            else:
                d += 1
        else:
            if c1 == y:
                b += 1
            else:
                c += 1

    return a, b, c, d


def mcnemar(a, b, c, d, alpha=0.05, onetailed=False, verbose=False):
    """Performs a mcnemar test.

    Parameters
    ----------
    a, b, c, d : ints in the form:
    A    B  A+B
    C    D  C+D
    A+C  B+D  n

    alpha: float
        Level of significance

    onetailed: bool
        False for two-tailed test
        True for one-tailed test

    Returns
    -------
    True if Null hypotheses pi1 == pi2 is accepted
    else False.
    """
    tot = float(a + b + c + d)

    try:
        z = (b - c)/ sqrt(b + c)
    except ZeroDivisionError as zd:
        return True

    if verbose:
    #    print "McNemar Test with A,B,C,D = ", A,B, C,D
    #    print "Ratios:p1, p2 = ",(A+B)/tot, (C + D) /tot
        print("Z test statistic Z = {}".format(z))

    if onetailed:
        if (b - c) > 0:
            zcrit2 = stats.norm.ppf(1-alpha)
            result = True if (z < zcrit2)else False
            #if verbose:
            #   print "Upper critical value=", zcrit2
            #   print "Decision:",  "Accept " if (result) else "Reject ",
            #   print "Null hypothesis at alpha = ", alpha
        else:
            zcrit1 = stats.norm.ppf(alpha)
            result = True if (z < zcrit1) else True
            #if verbose:
            #   print "Lower critical value=", zcrit1
            #   print "Decision:",  "Accept " if (result) else "Reject ",
            #   print "Null hypothesis at alpha = ", alpha

    else:
        zcrit1 = stats.norm.ppf(alpha/2.0)
        zcrit2 = stats.norm.ppf(1-alpha/2.0)
 
        result = True if (zcrit1 < z < zcrit2) else False
        if verbose:
            print("Lower and upper critical limits:", zcrit1, zcrit2)
            print("Decision:", "Accept " if result else "Reject ")
            print("Null hypothesis at alpha = ", alpha)
 
    return result


def mcnemar_test(y_vals, c1_preds, c2_preds, alpha=0.1, onetailed=True):
    """Returns the boolean value of the McNemar's Test of c1 vs. c2.

    Parameters
    ----------
    y_vals : np.ndarray or list
    Classification target values

    c1_preds : np.ndarray or list
    First classifier predictions

    c2_preds: np.ndarray or list
    Second classifier values

    alpha : float
    Level of significance
    
    onetailed: bool
                False for two-tailed test
                True for one-tailed test 

    Returns
    -------
    boolean : True if Null hypotheses pi1 == pi2 is accepted
              else False.
    """
    a, b, c, d = get_mcnemar_abcd(y_vals, c1_preds, c2_preds)
    return mcnemar(a, b, c, d, alpha=alpha, onetailed=onetailed, 
                   verbose=False)


#prep_mcnemar_borja
def entrenar_clasificador():
    from sklearn import tree

    # Entrenar clasificador
    out = np.array([1,1,2,2,3,3,4,4])
    inp = np.matrix('1 1 1; 0 0 0; 1 0 0;0 1 1;0 0 1;1 1 0;0 1 0;1 0 1')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(inp, out)
    return clf


def prep_mcnemar_borja(clf, y_vals, c1_preds, c2_preds):

    res = clf.predict(np.array([y_vals, c1_preds, c2_preds]).T)
    return sum(res==1), sum(res==3), sum(res==4), sum(res==2)


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

    y = np.array([1]*1000 + [0]*1000)
    c1 = np.array([1]*500 + [0]*500 + [1]*500 + [0]*500)
    c2 = np.array([0]*500 + [1]*500 + [0]*500 + [1]*500)
    
    a, b, c, d = prep_mcnemar_alex(y, c1, c2)
    a, b, c, d = prep_mcnemar_borja(clf, y, c1, c2)
    
    c2 = np.array([0]*250 + [1]*250 + [0]*250 + [1]*250 + [0]*250 + [1]*250 + [0]*250 + [1]*250)
