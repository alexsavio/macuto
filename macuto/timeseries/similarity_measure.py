
import abc
import numpy as np

# class Pizza(object):
#     def __init__(self, ingredients):
#         self.ingredients = ingredients
#
#     @classmethod
#     def from_fridge(cls, fridge):
#         return cls(fridge.get_cheese() + fridge.get_vegetables())

class TimeSeriesGroupMeasure(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def fit(ts_set1, ts_set2):
        """
        Returns the group measure value between both
        sets of timeseries.

        @param ts_set1: ndarray
        n_timeseries x ts_size

        @param ts_set2: ndarray
        n_timeseries x ts_size

        @return:
        """


class CorrelationMeasure(TimeSeriesGroupMeasure):

    def __init__(self):
        pass

    @staticmethod
    def fit(ts_set1, ts_set2, lb=0, ub=None, **kwargs):
        """
        Returns a list of correlation values between the time series in ts_set1
        and ts_set2.

        @param ts_set1: ndarray
        Time series matrix: n_samps x time_size

        @param ts_set2: ndarray
        Time series matrix: n_samps x time_size

        @param lb: float (optional)
        @param ub: float (optional)
        Lower and upper band of a pass-band into which the data will be
        filtered. Default: lb=0, ub=None (max frequency)
        Define a frequency band of interest

        @param kwargs:

        @return: list
        List of correlation values between each ts_set1 timeseries and the
        timeseries in ts_set2
        """
        import nitime.analysis as nta

        analyzer = nta.SeedCorrelationAnalyzer(ts_set1, ts_set2, lb=lb, ub=ub)

        n_seeds = ts_set1.data.shape[0] if ts_set1.data.ndim > 1 else 1
        if n_seeds == 1:
            return analyzer.corrcoef
        else:
            cor = []
            for seed in range(n_seeds):
                cor.append(analyzer.corrcoef[seed])

        return cor


class MeanCorrelationMeasure(CorrelationMeasure):

    def __init__(self):
        CorrelationMeasure.__init__(self)

    @staticmethod
    def fit(ts_set1, ts_set2, lb=0, ub=None, **kwargs):
        """
        Returns the mean correlation value of all coherences the time series in ts_set1
        and ts_set2.

        @param ts_set1: ndarray
        Time series matrix: n_samps x time_size

        @param ts_set2: ndarray
        Time series matrix: n_samps x time_size

        @param lb: float (optional)
        @param ub: float (optional)
        Lower and upper band of a pass-band into which the data will be
        filtered. Default: lb=0, ub=None (max frequency)
        Define a frequency band of interest

        @param kwargs:
        'NFFT'

        @return: list
        List of correlation values between each ts_set1 timeseries and the
        timeseries in ts_set2
        """
        cor = CorrelationMeasure.fit(ts_set1, ts_set2, lb=lb, ub=ub, **kwargs)
        return np.mean(cor)


class CoherenceMeasure(TimeSeriesGroupMeasure):

    def __init__(self):
        pass

    @staticmethod
    def fit(ts_set1, ts_set2, lb=0, ub=None, **kwargs):
        """
        Returns a list of coherence values between the time series in ts_set1 
        and ts_set2.

        @param ts_set1: ndarray
        Time series matrix: n_samps x time_size

        @param ts_set2: ndarray
        Time series matrix: n_samps x time_size

        @param lb: float (optional)
        @param ub: float (optional)
        Lower and upper band of a pass-band into which the data will be
        filtered. Default: lb=0, ub=None (max frequency)
        Define a frequency band of interest

        @param kwargs:
        'NFFT'

        @return: list
        List of correlation values between each ts_set1 timeseries and the
        timeseries in ts_set2
        """

        import nitime.analysis as nta

        if 'NFFT' in kwargs:
            fft_par = kwargs['NFFT']
            analyzer = nta.SeedCoherenceAnalyzer(ts_set1, ts_set2, lb=lb, ub=ub, method={'NFFT': kwargs['NFFT']})
        else:
            analyzer = nta.SeedCoherenceAnalyzer(ts_set1, ts_set2, lb=lb, ub=ub)

        n_seeds = ts_set1.data.shape[0] if ts_set1.data.ndim > 1 else 1
        if n_seeds == 1:
            return np.mean(analyzer.coherence, -1)
        else:
            coh = []
            for seed in range(n_seeds):
                # Averaging on the last dimension
                coh.append(np.mean(analyzer.coherence[seed], -1))

        return coh


class MeanCoherenceMeasure(CoherenceMeasure):

    def __init__(self):
        CoherenceMeasure.__init__(self)

    @staticmethod
    def fit(ts_set1, ts_set2, lb=0, ub=None, **kwargs):
        """
        Returns the mean coherence value of all coherences the time series in ts_set1
        and ts_set2.

        @param ts_set1: ndarray
        Time series matrix: n_samps x time_size

        @param ts_set2: ndarray
        Time series matrix: n_samps x time_size

        @param lb: float (optional)
        @param ub: float (optional)
        Lower and upper band of a pass-band into which the data will be
        filtered. Default: lb=0, ub=None (max frequency)
        Define a frequency band of interest

        @param kwargs:
        'NFFT'

        @return: list
        List of correlation values between each ts_set1 timeseries and the
        timeseries in ts_set2
        """
        coh = CoherenceMeasure.fit(ts_set1, ts_set2, lb=lb, ub=ub, **kwargs)
        return np.mean(coh)


class CrossCorrelationMeasure(TimeSeriesGroupMeasure):

    def __init__(self):
        pass

    @staticmethod
    def fit(ts_set1, ts_set2, lb=0, ub=None, **kwargs):
        """
        Returns a matrix with pearson correlation between pairs all time series in both sets.
        #---------------------------------------------------------------------------
        Inputs
        ------
        ts_set1        : Time series matrix: n_samps x time_size
        ts_set2        : Time series matrix: n_samps x time_size

        Outputs
        -------
        Matrix (n_samps1 x n_samps2) correlation values between each 
        ts_set1 timeseries and the timeseries in ts_set2
        """
        from scipy.stats import pearsonr

        n1 = ts_set1.shape[0]
        n2 = ts_set2.shape[0]

        mp = np.array(n1, n2)

        for i1 in range(n1):
            t1 = ts_set1[i1, :]
            for i2 in range(n2):
                t2 = ts_set2[i2, :]
                mp[i1, i2] = pearsonr(t1, t2)

        return mp


class SimilarityMeasureFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def create_method(method_name):
        """
        Returns a TimeSeriesGroupMeasure class given its name.

        @param method_name: string
        Choices: 'crosscorrelation', 'correlation', 'coherence',
        'mean_coherence', 'mean_correlation'

        @return:
        Timeseries selection method function

        @note: See: http://nipy.org/nitime/examples/seed_analysis.html
        for more information
        """
        measure_class = CorrelationMeasure
        if method_name == 'crosscorrelation' : measure_class =  CrossCorrelationMeasure
        if method_name == 'correlation'      : measure_class =       CorrelationMeasure
        if method_name == 'coherence'        : measure_class =         CoherenceMeasure
        if method_name == 'mean_coherence'   : measure_class =     MeanCoherenceMeasure
        if method_name == 'mean_correlation' : measure_class =   MeanCorrelationMeasure

        return measure_class.fit()

