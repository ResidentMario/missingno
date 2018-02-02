"""
Utilities test module. Asserts that utility functions are correct.
"""

import unittest
import pandas as pd
import numpy as np

import sys; sys.path.append("../")
import missingno as msno


class TestNullitySort(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan]})

    def test_no_op(self):
        expected = self.df
        result = msno.nullity_sort(self.df, sort=None)

        assert result.equals(expected)

    def test_ascending_sort(self):
        result = msno.nullity_sort(self.df, sort='ascending')
        expected = self.df.iloc[[2, 1, 0]]
        assert result.equals(expected)

    def test_descending_sort(self):
        result = msno.nullity_sort(self.df, sort='descending')
        expected = self.df.iloc[[0, 1, 2]]
        assert result.equals(expected)


class TestNullityFilter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan], 'C': [0, 0, 0]})

    def test_no_op(self):
        assert self.df.equals(msno.nullity_filter(self.df))
        assert self.df.equals(msno.nullity_filter(self.df, filter='top'))
        assert self.df.equals(msno.nullity_filter(self.df, filter='bottom'))

    def test_percentile_cutoff_top_p(self):
        expected = self.df.loc[:, ['B', 'C']]
        result = msno.nullity_filter(self.df, p=0.6, filter='top')
        assert result.equals(expected)

    def test_percentile_cutoff_bottom_p(self):
        expected = self.df.loc[:, ['A']]
        result = msno.nullity_filter(self.df, p=0.6, filter='bottom')
        assert result.equals(expected)

    def test_percentile_cutoff_bottom_n(self):
        expected = self.df.loc[:, ['C']]
        result = msno.nullity_filter(self.df, n=1, filter='top')
        assert result.equals(expected)

    def test_percentile_cutoff_top_n(self):
        expected = self.df.loc[:, ['A']]
        result = msno.nullity_filter(self.df, n=1, filter='bottom')
        assert result.equals(expected)

    def test_combined_cutoff_top(self):
        expected = self.df.loc[:, ['C']]
        result = msno.nullity_filter(self.df, n=2, p=0.7, filter='top')
        assert result.equals(expected)

    def test_combined_cutoff_bottom(self):
        expected = self.df.loc[:, ['A']]
        result = msno.nullity_filter(self.df, n=2, p=0.4, filter='bottom')
        assert result.equals(expected)
