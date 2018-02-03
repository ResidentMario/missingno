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
#     def test_simple_matrix(self):
#         return msno.matrix(self.simple_df, inline=False)
#
#     @pytest.mark.mpl_image_compare
#     def test_no_sparkline_matrix(self):
#         return msno.matrix(self.simple_df, inline=False, sparkline=False)
#
#     @pytest.mark.mpl_image_compare
#     def test_width_ratios_matrix(self):
#         return msno.matrix(self.simple_df, inline=False, width_ratios=(30, 1))
#
#     @pytest.mark.mpl_image_compare
#     def test_color_matrix(self):
#         return msno.matrix(self.simple_df, inline=False, color=(70 / 255, 130 / 255, 180 / 255))
#
#     @pytest.mark.mpl_image_compare
#     def test_fontsize_matrix(self):
#         return msno.matrix(self.simple_df, inline=False, fontsize=8)
#
#     @pytest.mark.mpl_image_compare
#     def test_freq_matrix(self):
#         return msno.matrix(self.freq_df, inline=False, freq='BQ')
#
#     @pytest.mark.mpl_image_compare
#     def test_large_matrix(self):
#         return msno.matrix(self.large_df, inline=False)


# class TestBar(unittest.TestCase):
#     """
#     Bar chart visualizations look very visually different between the savefig backend and the default notebook backend.
#     """
#     def setUp(self):
#         np.random.seed(42)
#         self.simple_df = pd.DataFrame((np.random.random((20, 10)) > 0.5), columns=range(0, 10)).replace(False, np.nan)
#
#     @pytest.mark.mpl_image_compare
#     def test_simple_bar(self):
#         return msno.bar(self.simple_df, inline=False)
#
#     @pytest.mark.mpl_image_compare
#     def test_log_bar(self):
#         return msno.bar(self.simple_df, log=True, inline=False)


# class TestHeatmap(unittest.TestCase):
#     def setUp(self):
#         np.random.seed(42)
#         self.simple_df = pd.DataFrame((np.random.random((20, 10)) > 0.5), columns=range(0, 10)).replace(False, np.nan)
#         self.large_df = pd.DataFrame((np.random.random((250, 60)) > 0.5)).replace(False, np.nan)
#
#     @pytest.mark.mpl_image_compare
#     def test_simple_heatmap(self):
#         return msno.heatmap(self.simple_df, inline=False)
#
#     @pytest.mark.mpl_image_compare
#     def test_unlabelled_heatmap(self):
#         return msno.heatmap(self.simple_df, labels=False, inline=False)
#
#     @pytest.mark.mpl_image_compare
#     def test_alternative_colormap_heatmap(self):
#         return msno.heatmap(self.simple_df, cmap='viridis', inline=False)


class TestDendrogram(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        simple_df = pd.DataFrame((np.random.random((20, 10))), columns=range(0, 10))
        simple_df.iloc[:, :2] = (simple_df.iloc[:, :2] > 0.2)
        simple_df.iloc[:, 2:5] = (simple_df.iloc[:, 2:5] > 0.8)
        simple_df.iloc[:, 5:10] = (simple_df.iloc[:, 2:5] > 0.5)
        self.simple_df = simple_df.replace(False, np.nan)

    @pytest.mark.mpl_image_compare
    def test_simple_dendrogram(self):
        return msno.dendrogram(self.simple_df, inline=False)

    @pytest.mark.mpl_image_compare
    def test_orientation_dendrogram(self):
        return msno.dendrogram(self.simple_df, orientation='right', inline=False)

    @pytest.mark.mpl_image_compare
    def test_method_dendrogram(self):
        return msno.dendrogram(self.simple_df, method='single', inline=False)
