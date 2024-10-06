"""Utility functions for missingno."""
import numpy as np
import narwhals as nw

def pearson_corr(col1, col2):
    mean1 = col1.mean()
    mean2 = col2.mean()
    deviation1 = col1 - mean1
    deviation2 = col2 - mean2
    product_of_deviations = deviation1 * deviation2
    sum_of_products = product_of_deviations.sum()
    std1 = col1.std()
    std2 = col2.std()
    n = len(col1)
    return sum_of_products / ((n - 1) * std1 * std2)

@nw.narwhalify
def nullity_sort(df, sort=None, axis='columns'):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.

    :param df: The DataFrame object being sorted.
    :param sort: The sorting method: either "ascending", "descending", or None (default).
    :return: The nullity-sorted DataFrame.
    """
    return df
    # if sort is None:
    #     return df
    # elif sort not in ['ascending', 'descending']:
    #     raise ValueError('The "sort" parameter must be set to "ascending" or "descending".')

    # if axis not in ['rows', 'columns']:
    #     raise ValueError('The "axis" parameter must be set to "rows" or "columns".')

    # if axis == 'columns':
    #     if sort == 'ascending':
    #         return df.iloc[np.argsort(df.count(axis='columns').values), :]
    #     elif sort == 'descending':
    #         return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]
    # elif axis == 'rows':
    #     if sort == 'ascending':
    #         return df.iloc[:, np.argsort(df.count(axis='rows').values)]
    #     elif sort == 'descending':
    #         return df.iloc[:, np.flipud(np.argsort(df.count(axis='rows').values))]

@nw.narwhalify
def nullity_filter(df, filter=None, p=0, n=0):
    """
    Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
    percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
    to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
    `nullity_filter(df, filter='top', p=.75, n=5)`.

    :param df: The DataFrame whose columns are being filtered.
    :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
    or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
    as None.
    :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
    completeness. Input should be in the range [0, 1].
    :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
    :return: The nullity-filtered `DataFrame`.
    """
    return df
    # if filter == 'top':
    #     if p:
    #         df = df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]
    #     if n:
    #         df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]
    # elif filter == 'bottom':
    #     if p:
    #         df = df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]
    #     if n:
    #         df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]
    # return df
