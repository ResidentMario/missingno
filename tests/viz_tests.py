"""
Visualization test module. Asserts that visualization functions work properly.
"""

import unittest
import pandas as pd
import numpy as np
import pytest

import sys
sys.path.append("../")

import missingno as msno
import matplotlib.pyplot as plt


class TestMatrix(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.simple_df = pd.DataFrame((np.random.random((20, 10)) > 0.5), columns=range(0, 10)).replace(False, np.nan)
        np.random.seed(42)
        self.freq_df = (
            pd.DataFrame((np.random.random(1000).reshape((50, 20)) > 0.5))
                .replace(False, np.nan)
                .set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M'))
        )
        np.random.seed(42)
        self.large_df = pd.DataFrame((np.random.random((250, 60)) > 0.5)).replace(False, np.nan)

    @pytest.mark.mpl_image_compare
    def test_simple_matrix(self):
        msno.matrix(self.simple_df)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_no_sparkline_matrix(self):
        msno.matrix(self.simple_df, sparkline=False)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_width_ratios_matrix(self):
        msno.matrix(self.simple_df, width_ratios=(30, 1))
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_color_matrix(self):
        msno.matrix(self.simple_df, color=(70 / 255, 130 / 255, 180 / 255))
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_fontsize_matrix(self):
        msno.matrix(self.simple_df, fontsize=8)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_freq_matrix(self):
        msno.matrix(self.freq_df, freq='BQ')
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_large_matrix(self):
        msno.matrix(self.large_df)
        return plt.gcf()


class TestBar(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.simple_df = pd.DataFrame((np.random.random((20, 10)) > 0.5), columns=range(0, 10)).replace(False, np.nan)

    @pytest.mark.mpl_image_compare
    def test_simple_bar(self):
        msno.bar(self.simple_df)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_log_bar(self):
        msno.bar(self.simple_df, log=True)
        return plt.gcf()


class TestHeatmap(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.simple_df = pd.DataFrame((np.random.random((20, 10)) > 0.5), columns=range(0, 10)).replace(False, np.nan)
        self.large_df = pd.DataFrame((np.random.random((250, 60)) > 0.5)).replace(False, np.nan)

    @pytest.mark.mpl_image_compare
    def test_simple_heatmap(self):
        msno.heatmap(self.simple_df)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_unlabelled_heatmap(self):
        msno.heatmap(self.simple_df, labels=False)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_alternative_colormap_heatmap(self):
        msno.heatmap(self.simple_df, cmap='viridis')
        return plt.gcf()


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
        msno.dendrogram(self.simple_df)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_orientation_dendrogram(self):
        msno.dendrogram(self.simple_df, orientation='right')
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_method_dendrogram(self):
        msno.dendrogram(self.simple_df, method='single')
        return plt.gcf()
