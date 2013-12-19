
import numpy as np


class CorrelationMeasure(object):

    def __init__(self):
        pass

    @staticmethod
    def fit(ts_set1, ts_set2, lb=0, ub=None, **kwargs):
        """
        Returns a list of coherence values between the time series in ts_set1 
        and ts_set2.
        See: http://nipy.org/nitime/examples/seed_analysis.html
        #---------------------------------------------------------------------------
        Inputs
        ------
        ts_set1        : Time series matrix: n_samps x time_size
        ts_set2        : Time series matrix: n_samps x time_size

        Outputs
        -------
        List of coherence values between each ts_set1 timeseries and the 
        timeseries in ts_set2
        """

        import nitime.analysis as nta

        analyzer = nta.SeedCorrelationAnalyzer(ts_set1, ts_set2)

        f_lb = lb
        f_ub = ub
        if f_lb and f_ub:
            freq_idx = np.where((analyzer.frequencies > f_lb) * (analyzer.frequencies < f_ub))[0]
        elif f_lb:
            freq_idx = np.where(analyzer.frequencies > f_lb)[0]
        elif f_ub:
            freq_idx = np.where(analyzer.frequencies < f_ub)[0]

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
    def fit(ts_set1, ts_set2, **kwargs):
        cor = CorrelationMeasure.fit(ts_set1, ts_set2, **kwargs)
        return np.mean(cor)


class CoherenceMeasure(object):

    def __init__(self):
        pass

    @staticmethod
    def fit(ts_set1, ts_set2, **kwargs):
        """
        Returns a list of coherence values between the time series in ts_set1 
        and ts_set2.
        See: http://nipy.org/nitime/examples/seed_analysis.html
        #---------------------------------------------------------------------------
        Inputs
        ------
        ts_set1        : Time series matrix: n_samps x time_size
        ts_set2        : Time series matrix: n_samps x time_size

        Kwargs
        ------
        lb,ub: float (optional)
        Lower and upper band of a pass-band into which the data will be
        filtered. Default: 0, Nyquist

        Outputs
        -------
        List of coherence values between each ts_set1 timeseries and the 
        timeseries in ts_set2
        """

        import nitime.analysis as nta

        if 'NFFT' in kwargs:
            fft_par = kwargs['NFFT']
            analyzer = nta.SeedCoherenceAnalyzer(ts_set1, ts_set2, method={'NFFT': kwargs['NFFT']})
        else:
            analyzer = nta.SeedCoherenceAnalyzer(ts_set1, ts_set2)

        f_lb = kwargs.get('lb', 0)
        f_ub = kwargs.get('ub', None)
        if f_lb and f_ub:
            freq_idx = np.where((analyzer.frequencies > f_lb) * (analyzer.frequencies < f_ub))[0]
        elif f_lb:
            freq_idx = np.where(analyzer.frequencies > f_lb)[0]
        elif f_ub:
            freq_idx = np.where(analyzer.frequencies < f_ub)[0]

        n_seeds = ts_set1.data.shape[0] if ts_set1.data.ndim > 1 else 1
        if n_seeds == 1: return np.mean(analyzer.coherence[:, freq_idx], -1)
        else:
            coh = []
            for seed in range(n_seeds):
                coh.append(np.mean(analyzer.coherence[seed][:, freq_idx], -1)) # Averaging on the
                                                                               # last dimension

        return coh


class MeanCoherenceMeasure(CoherenceMeasure):

    def __init__(self):
        CoherenceMeasure.__init__(self)

    @staticmethod
    def fit(ts_set1, ts_set2, **kwargs):
        coh = CoherenceMeasure.fit(ts_set1, ts_set2, **kwargs)
        return np.mean(coh)


class CrossCorrelationMeasure(object):

    def __init__(self):
        pass

    @staticmethod
    def fit(ts_set1, ts_set2, **kwargs):
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

    @classmethod
    def create_method(cls, method_name):
        """
        Returns a Timeseries selection method class given its name.
        Inputs
        ------
        method_name: a string
        'crosscorrelation', 'correlation', 'coherence', 
        'mean_coherence', 'mean_correlation'

        See: http://nipy.org/nitime/examples/seed_analysis.html

        Outputs
        -------
        Timeseries selection method class, use its fit() method.
        """
        if method_name == 'crosscorrelation' : return CrossCorrelationMeasure()
        if method_name == 'correlation'      : return      CorrelationMeasure()
        if method_name == 'coherence'        : return        CoherenceMeasure()
        if method_name == 'mean_coherence'   : return    MeanCoherenceMeasure()
        if method_name == 'mean_correlation' : return  MeanCorrelationMeasure()

        return     CorrelationMeasure()

