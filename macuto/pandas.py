
import pandas as pd
from collections import OrderedDict

def count_nan(df):
    """
    Counts the number of NaN occurrences in each column of df.

    :param df: pd.DataFrame

    :returns: dict
    A dictionary with the NaN count for each column, keyed by the column names.
    """
    nan_counts = OrderedDict()
    for col in df.columns:
        nan_counts[col] = pd.isnull(df[col].values).sum()
    return nan_counts
