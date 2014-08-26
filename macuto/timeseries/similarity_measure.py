
import numpy as np
import nitime
import nitime.analysis as nta

#TODO: Give the option here to use full matrices, instead of just ts_sets

#See Ledoit-Wolf covariance estimation and Graph-Lasso
#http://www.sciencedirect.com/science/article/pii/S1053811910011602
#http://nilearn.github.io/data_analysis/functional_connectomes.html
#http://scikit-learn.org/stable/modules/covariance.html
#http://nilearn.github.io/developers/group_sparse_covariance.html


class TimeSeriesGroupMeasure(object):
    """
    A strategy class to use any of the time series
    group measures methods given as a callable objet.
    """
    def __init__(self, algorithm):
        """
        TimeSeriesGroupMeasure, Strategy class constructor.

        :param algorithm: callable object
        """
        self.algorithm = algorithm
        self.measure_value = None

    def fit_transform(self, ts_set1, ts_set2, **kwargs):
        """
        Returns the group measure value between both
        sets of timeseries.

        @param ts_set1: nitime.TimeSeries
        n_timeseries x ts_size

        @param ts_set2: nitime.TimeSeries
        n_timeseries x ts_size

        @return:
        """
        self.measurer = self.algorithm(ts_set1, ts_set2, **kwargs)
        return self.measurer.measure_value


class NiCorrelationMeasure(object):
    """
    Calculates the correlation using nitime.

    It's a redundant implementation, I haven't compared them yet
    """
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        """
        Returns a the Pearson's Correlation value between all time series in both sets.

        @param ts_set1: Time series nitime.TimeSeries: n_samps x time_size
        @param ts_set2: Time series nitime.TimeSeries: n_samps x time_size

        @param lb: float
        Frequency lower bound

        @param ub: float
        Frequency upper bound

        @return: float
        Scalar float value
        """
        if ts_set1.data.ndim == 1:
            ts_data = np.concatenate([[ts_set1.data, ts_set2.data]])
        else:
            ts_data = np.concatenate([ts_set1.data, ts_set2.data])

        ts = nitime.TimeSeries(ts_data, sampling_interval=TR)

        corr = nta.CorrelationAnalyzer(ts)

        self.measure_value = corr.corrcoef[0, 1]


class NiCoherenceMeasure(object):
    """
    """
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        """
        Returns a the Spectral Coherence value between all time series in both sets.

        @param ts_set1: Time series nitime.TimeSeries: n_samps x time_size
        @param ts_set2: Time series nitime.TimeSeries: n_samps x time_size

        @param lb: float
        Frequency lower bound

        @param ub: float
        Frequency upper bound

        @return: float
        Scalar float value
        """
        if ts_set1.data.ndim == 1:
            ts_data = np.concatenate([[ts_set1.data, ts_set2.data]])
        else:
            ts_data = np.concatenate([ts_set1.data, ts_set2.data])

        ts = nitime.TimeSeries(ts_data, sampling_interval=TR)

        coh = nta.CoherenceAnalyzer(ts)

        freq_idx_coh = np.where((coh.frequencies > lb) * (coh.frequencies < ub))[0]

        self.measure_value = np.mean(coh.coherence[:, :, freq_idx_coh], -1)  # Averaging on the last dimension


class NiGrangerCausalityMeasure(object):
    """
    """
    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        """
        Returns a the Granger Causality value all time series in both sets.

        @param ts_set1: Time series nitime.TimeSeries: n_samps x time_size
        @param ts_set2: Time series nitime.TimeSeries: n_samps x time_size

        @param lb: float
        Frequency lower bound

        @param ub: float
        Frequency upper bound

        @return: float
        Scalar float value
        """
        if ts_set1.data.ndim == 1:
            ts_data = np.concatenate([[ts_set1.data, ts_set2.data]])
        else:
            ts_data = np.concatenate([ts_set1.data, ts_set2.data])

        ts = nitime.TimeSeries(ts_data, sampling_interval=TR)

        gc = nta.GrangerAnalyzer(ts, order=1)

        freq_idx_gc = np.where((gc.frequencies > lb) * (gc.frequencies < ub))[0]

        self.measure_value = np.mean(gc.causality_xy[:, :, freq_idx_gc], -1)


class CorrelationMeasure(object):

    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        """
        Returns a matrix with pearson correlation between pairs all time series in both sets.
        #---------------------------------------------------------------------------
        Inputs
        ------
        ts_set1        : Time series nitime.TimeSeries: n_samps x time_size
        ts_set2        : Time series nitime.TimeSeries: n_samps x time_size

        Outputs
        -------
        Matrix (n_samps1 x n_samps2) correlation values between each
        ts_set1 timeseries and the timeseries in ts_set2
        """
        from scipy.stats import pearsonr

        ts1 = ts_set1.data if hasattr(ts_set1, 'data') else ts_set1
        ts2 = ts_set2.data if hasattr(ts_set2, 'data') else ts_set2

        n1 = ts1.shape[0]
        n2 = ts2.shape[0]
        if (ts1.ndim > 1 or ts2.ndim > 1) and n1 == n2 > 1:
            mp = np.array(n1, n2)
            for i1 in list(range(n1)):
                t1 = ts1[i1, :]
                for i2 in list(range(n2)):
                    t2 = ts2[i2, :]
                    mp[i1, i2] = pearsonr(t1, t2)
            self.measure_value = mp

        else:
            self.measure_value = pearsonr(ts1.flatten(), ts2.flatten())[0]




#class GrangerCausalityMeasure(OneVsOneTimeSeriesMeasure):
#class MutualInformationMeasure(OneVsOneTimeSeriesMeasure):

#----------------------------------------------------------------------------
#Many Vs Many Time Series Measures
#----------------------------------------------------------------------------
class SeedCorrelationMeasure(object):

    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        """
        Returns a list of correlation values between the time series in ts_set1
        and ts_set2.

        @param ts_set1: nitime.TimeSeries
        Time series matrix: n_samps x time_size

        @param ts_set2: nitime.TimeSeries
        Time series matrix: n_samps x time_size

        @param kwargs:

        @return: list
        List of correlation values between each ts_set1 timeseries and the
        timeseries in ts_set2
        """
        import nitime.analysis as nta

        analyzer = nta.SeedCorrelationAnalyzer(ts_set1, ts_set2)

        n_seeds = ts_set1.data.shape[0] if ts_set1.data.ndim > 1 else 1
        if n_seeds == 1:
            cor = analyzer.corrcoef
        else:
            cor = []
            for seed in range(n_seeds):
                cor.append(analyzer.corrcoef[seed])

        self.measure_value = cor


class MeanSeedCorrelationMeasure(SeedCorrelationMeasure):

    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        """
        Returns the mean correlation value of all seed correlations the time series in ts_set1
        and ts_set2.

        @param ts_set1: nitime.TimeSeries
        Time series matrix: n_samps x time_size

        @param ts_set2: nitime.TimeSeries
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
        super(MeanSeedCorrelationMeasure, self).__init__(ts_set1, ts_set2,
                                                         lb=lb, ub=ub, TR=2,
                                                         **kwargs)
        self.measure_value = np.mean(self.measure_value)


class SeedCoherenceMeasure(object):

    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        """
        Returns a list of coherence values between the time series in ts_set1 
        and ts_set2.

        @param ts_set1: nitime.TimeSeries
        Time series matrix: n_samps x time_size

        @param ts_set2: nitime.TimeSeries
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

        fft_par = kwargs.pop('NFFT', None)
        if fft_par is not None:
            analyzer = nta.SeedCoherenceAnalyzer(ts_set1, ts_set2, lb=lb, ub=ub,
                                                 method={'NFFT': fft_par})
        else:
            analyzer = nta.SeedCoherenceAnalyzer(ts_set1, ts_set2, lb=lb, ub=ub)

        n_seeds = ts_set1.data.shape[0] if ts_set1.data.ndim > 1 else 1
        if n_seeds == 1:
            coh = np.mean(analyzer.coherence, -1)
        else:
            coh = []
            for seed in range(n_seeds):
                # Averaging on the last dimension
                coh.append(np.mean(analyzer.coherence[seed], -1))

        self.measure_value = coh


class MeanSeedCoherenceMeasure(SeedCoherenceMeasure):

    def __init__(self, ts_set1, ts_set2, lb=0, ub=None, TR=2, **kwargs):
        """
        Returns the mean coherence value of all seed coherences the time series in ts_set1
        and ts_set2.

        @param ts_set1: nitime.TimeSeries
        Time series matrix: n_samps x time_size

        @param ts_set2: nitime.TimeSeries
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
        super(MeanSeedCoherenceMeasure, self).__init__(ts_set1, ts_set2,
                                                         lb=lb, ub=ub, TR=2,
                                                         **kwargs)
        self.measure_value = np.mean(self.measure_value)


class SimilarityMeasureFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def create_method(method_name):
        """
        Returns a TimeSeriesGroupMeasure class given its name.

        @param method_name: string
        Choices:
        'correlation', 'coherence', 'nicorrelation', 'grangercausality'
        'seedcorrelation', 'seedcoherence', 'mean_seedcoherence', 'mean_seedcorrelation'

        @return:
        Timeseries selection method function

        @note: See: http://nipy.org/nitime/examples/seed_analysis.html
        for more information
        """

        algorithm = CorrelationMeasure
        if method_name == 'correlation'        : algorithm = CorrelationMeasure
        if method_name == 'coherence'          : algorithm = NiCoherenceMeasure
        if method_name == 'grangercausality'   : algorithm = NiGrangerCausalityMeasure
        if method_name == 'nicorrelation'      : algorithm = NiCorrelationMeasure

        if method_name == 'seedcorrelation'        : algorithm = SeedCorrelationMeasure
        if method_name == 'seedcoherence'          : algorithm = SeedCoherenceMeasure
        if method_name == 'mean_seedcoherence'     : algorithm = MeanSeedCoherenceMeasure
        if method_name == 'mean_seedcorrelation'   : algorithm = MeanSeedCorrelationMeasure

        #if method_name == 'mutual_information' : algorithm = MutualInformationMeasure
        #if method_name == 'granger_causality'  : algorithm = GrangerCausalityMeasure

        return TimeSeriesGroupMeasure(algorithm)


# import os
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.mlab import csv2rec
#
# import nitime
# import nitime.analysis as nta
# import nitime.timeseries as ts
# import nitime.utils as tsu
# from nitime.plotting import drawmatrix_channels
#
# TR = 1.89
# f_ub = 0.15
# f_lb = 0.02
#
# data_path = os.path.join(nitime.__path__[0], 'data')
#
# data_rec = csv2rec(os.path.join(data_path, 'fmri_timeseries.csv'))
#
# roi_names = np.array(data_rec.dtype.names)
# nseq = len(roi_names)
# n_samples = data_rec.shape[0]
# data = np.zeros((nseq, n_samples))
#
# for n_idx, roi in enumerate(roi_names):
#     data[n_idx] = data_rec[roi]
#
# G = nta.GrangerAnalyzer(time_series, order=1)
# C1 = nta.CoherenceAnalyzer(time_series)
# C2 = nta.CorrelationAnalyzer(time_series)
# freq_idx_G = np.where((G.frequencies > f_lb) * (G.frequencies < f_ub))[0]
# freq_idx_C = np.where((C1.frequencies > f_lb) * (C1.frequencies < f_ub))[0]
#
# coh = np.mean(C1.coherence[:, :, freq_idx_C], -1)  # Averaging on the last dimension
# g1 = np.mean(G.causality_xy[:, :, freq_idx_G], -1)
#
# fig01 = drawmatrix_channels(coh, roi_names, size=[10., 10.], color_anchor=0)
#
# fig02 = drawmatrix_channels(C2.corrcoef, roi_names, size=[10., 10.], color_anchor=0)
# g2 = np.mean(G.causality_xy[:, :, freq_idx_G] - G.causality_yx[:, :, freq_idx_G], -1)
# fig04 = drawmatrix_channels(g2, roi_names, size=[10., 10.], color_anchor=0)
