"""
Visualization test module. Asserts that visualization functions work properly.
"""

import unittest
import pandas as pd
import numpy as np
import pytest

import sys; sys.path.append("../")
import missingno as msno


# class TestMatrix(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(42)
#         self.simple_df = pd.DataFrame((np.random.random((20, 10)) > 0.5), columns=range(0, 10)).replace(False, np.nan)
#         np.random.seed(42)
#         self.freq_df = (
#             pd.DataFrame((np.random.random(1000).reshape((50, 20)) > 0.5))
#                 .replace(False, np.nan)
#                 .set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M'))
#         )
#         np.random.seed(42)
#         self.large_df = pd.DataFrame((np.random.random((250, 60)) > 0.5)).replace(False, np.nan)
#
#     @pytest.mark.mpl_image_compare
#     def test_simple(self):
#         return msno.matrix(self.simple_df, inline=False)
#
#     @pytest.mark.mpl_image_compare
#     def test_no_sparkline(self):
#         return msno.matrix(self.simple_df, inline=False, sparkline=False)
#
#     @pytest.mark.mpl_image_compare
#     def test_width_ratios(self):
#         return msno.matrix(self.simple_df, inline=False, width_ratios=(30, 1))
#
#     @pytest.mark.mpl_image_compare
#     def test_color(self):
#         return msno.matrix(self.simple_df, inline=False, color=(70 / 255, 130 / 255, 180 / 255))
#
#     @pytest.mark.mpl_image_compare
#     def test_fontsize(self):
#         return msno.matrix(self.simple_df, inline=False, fontsize=8)
#
#     @pytest.mark.mpl_image_compare
#     def test_freq(self):
#         return msno.matrix(self.freq_df, inline=False, freq='BQ')
#
#     @pytest.mark.mpl_image_compare
#     def test_large(self):
#         return msno.matrix(self.large_df, inline=False)


class TestBar(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.simple_df = pd.DataFrame((np.random.random((20, 10)) > 0.5), columns=range(0, 10)).replace(False, np.nan)

    @pytest.mark.mpl_image_compare
    def test_simple(self):
        return msno.bar(self.simple_df, inline=False)
