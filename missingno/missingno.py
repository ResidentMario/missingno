import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

__version__ = "0.3.4"


def _ascending_sort(df):
    """
    Helper method for sorting.
    Returns a DataFrame whose values have been rearranged by ascending completeness.
    """
    return df.iloc[np.argsort(df.count(axis='columns').values), :]


def _descending_sort(df):
    """
    Helper method for sorting.
    Returns a DataFrame whose values have been rearranged by descending completeness.
    """
    return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]


def nullity_sort(df, sort=None):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.

    :param df: The DataFrame object being sorted.
    :param sort: The sorting method: either "ascending", "descending", or None (default).
    :return: The nullity-sorted DataFrame.
    """
    _df = df
    if sort == "ascending":
        _df = _ascending_sort(df)
    elif sort == "descending":
        _df = _descending_sort(df)
    return _df


def _n_top_complete_filter(df, n):
    """
    Helper method for filtering a DataFrame.
    Returns the top n most populated entry columns.
    """
    return df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]


def _n_bottom_complete_filter(df, n):
    """
    Helper method for filtering a DataFrame.
    Returns the bottom n least populated entry columns.
    """
    return df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]


def _p_top_complete_filter(df, p):
    """
    Helper method for filtering a DataFrame.
    Returns the entry columns which are at least p*100 percent completeness.
    """
    return df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]


def _p_bottom_complete_filter(df, p):
    """
    Helper method for filtering a DataFrame.
    Returns the entry columns which are at most p*100 percent completeness.
    """
    return df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]


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
    _df = df
    if filter == "top":
        if p:
            _df = _p_top_complete_filter(_df, p)
        if n:
            _df = _n_top_complete_filter(_df, n)
    elif filter == "bottom":
        if p:
            _df = _p_bottom_complete_filter(_df, p)
        if n:
            _df = _n_bottom_complete_filter(_df, n)
    return _df


def matrix(df,
           filter=None, n=0, p=0, sort=None,
           figsize=(25, 10), width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
           fontsize=16, labels=None, sparkline=True, inline=True,
           freq=None):
    """
    Presents a `matplotlib` matrix visualization of the nullity of the given DataFrame.

    Note that for the default `figsize` 250 is a soft display limit: specifying a number of records greater than
    approximately this value will cause certain records to show up in the sparkline but not in the matrix, which can
    be confusing.


    The default vertical display will fit up to 50 columns. If more than 50 columns are specified and the labels
    parameter is left unspecified the visualization will automatically drop the labels as they will not be very
    readable. You can override this behavior using `labels=True` and your own `fontsize` parameter.

    :param df: The DataFrame whose completeness is being nullity matrix mapped.
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default). See
    `nullity_filter()` for more information.
    :param n: The cap on the number of columns to include in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param p: The cap on the percentage fill of the columns in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param sort: The sort to apply to the heatmap. Should be one of "ascending", "descending", or None. See
    `nullity_sort()` for more information.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter.
    For the vertical configuration this defaults to (20, 10); the horizontal configuration computes a sliding value
    by default based on the number of columns that need to be displayed.
    :param fontsize: The figure's font size. This default to 16.
    :param labels: Whether or not to display the column names. Would need to be turned off on particularly large
    displays. Defaults to True.
    :param sparkline: Whether or not to display the sparkline. Defaults to True.
    :param width_ratios: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15,
    1)`. Does nothing if `sparkline=False`.
    :param color: The color of the filled columns. Default is a medium dark gray: the RGB multiple `(0.25, 0.25, 0.25)`.
    :return: If `inline` is True, the underlying `matplotlib.figure` object. Else, nothing.
    """

    # Apply filters and sorts.
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort)

    height = df.shape[0]
    width = df.shape[1]

    # z is the color-mask array.
    z = df.notnull().values

    # g is a NxNx3 matrix
    g = np.zeros((height, width, 3))

    # Apply the z color-mask to set the RGB of each pixel.
    g[z < 0.5] = [1, 1, 1]
    g[z > 0.5] = color

    # Set up the matplotlib grid layout.
    # If the sparkline is removed the layout is a unary subplot.
    # If the sparkline is included the layout is a left-right subplot.
    fig = plt.figure(figsize=figsize)
    if sparkline:
        gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
        gs.update(wspace=0.08)
        ax1 = plt.subplot(gs[1])
    else:
        gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])

    # Create the nullity plot.
    ax0.imshow(g, interpolation='none')

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(b=False)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)

    # Set up and rotate the column ticks.
    # The labels argument is set to None by default. If the user specifies it in the argument,
    # respect that specification. Otherwise display for <= 50 columns and do not display for > 50.
    if labels or (labels is None and len(df.columns) <= 50):
        ha = 'left'
        ax0.set_xticks(list(range(0, width)))
        ax0.set_xticklabels(list(df.columns), rotation=45, ha=ha, fontsize=fontsize)
    else:
        ax0.set_xticks([])

    # Adds Timestamps ticks if freq is not None,
    # else sets up the two top-bottom row ticks.
    if freq:
        ts_list = []

        if type(df.index) == pd.tseries.period.PeriodIndex:
            ts_array = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))

        elif type(df.index) == pd.tseries.index.DatetimeIndex:
            ts_array = pd.date_range(df.index.date[0], df.index.date[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index.date[0], df.index.date[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))
        else:
            raise KeyError("Dataframe index must be PeriodIndex or DatetimeIndex.")
        try:
            for value in ts_array:
                ts_list.append(df.index.get_loc(value))
        except KeyError:
            raise KeyError("Could not divide time index into desired frequency.")

        ax0.set_yticks(ts_list)
        ax0.set_yticklabels(ts_ticks, fontsize=20, rotation=0)
    else:
        ax0.set_yticks([0, df.shape[0] - 1])
        ax0.set_yticklabels([1, df.shape[0]], fontsize=20, rotation=0)

    # Create the inter-column vertical grid.
    in_between_point = [x + 0.5 for x in range(0, width - 1)]
    for in_between_point in in_between_point:
        ax0.axvline(in_between_point, linestyle='-', color='white')

    if sparkline:
        # Calculate row-wise completeness for the sparkline.
        completeness_srs = df.notnull().astype(bool).sum(axis=1)
        x_domain = list(range(0, height))
        y_range = list(reversed(completeness_srs.values))
        min_completeness = min(y_range)
        max_completeness = max(y_range)
        min_completeness_index = y_range.index(min_completeness)
        max_completeness_index = y_range.index(max_completeness)

        # Set up the sparkline.
        ax1.grid(b=False)
        ax1.set_aspect('auto')
        ax1.set_facecolor((1, 1, 1))
        # Remove the black border.
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)

        # Plot sparkline---plot is sideways so the x and y axis are reversed.
        ax1.plot(y_range, x_domain, color=color)

        if labels:
            # Figure out what case to display the label in: mixed, upper, lower.
            label = 'Data Completeness'
            if df.columns[0].islower():
                label = label.lower()
            if df.columns[0].isupper():
                label = label.upper()

            # Set up and rotate the sparkline label.
            ha = 'left'
            ax1.set_xticks([min_completeness + (max_completeness - min_completeness) / 2])
            ax1.set_xticklabels([label], rotation=45, ha=ha, fontsize=fontsize)
            ax1.xaxis.tick_top()
            ax1.set_yticks([])
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])

        # Add maximum and minimum labels.
        ax1.annotate(max_completeness,
                     xy=(max_completeness, max_completeness_index),
                     xytext=(max_completeness + 2, max_completeness_index),
                     fontsize=14,
                     va='center',
                     ha='left')
        ax1.annotate(min_completeness,
                     xy=(min_completeness, min_completeness_index),
                     xytext=(min_completeness - 2, min_completeness_index),
                     fontsize=14,
                     va='center',
                     ha='right')

        # Add maximum and minimum circles.
        ax1.set_xlim([min_completeness - 2, max_completeness + 2])  # Otherwise the circles are cut off.
        ax1.plot([min_completeness], [min_completeness_index], '.', color=color, markersize=10.0)
        ax1.plot([max_completeness], [max_completeness_index], '.', color=color, markersize=10.0)

        # Remove tick mark (only works after plotting).
        ax1.xaxis.set_ticks_position('none')

    # Plot if inline, return the figure if not.
    if inline:
        plt.show()
    else:
        return plt


def bar(df, figsize=(24, 10), fontsize=16, labels=None, log=False, color=(0.25, 0.25, 0.25), inline=True,
        filter=None, n=0, p=0, sort=None):
    """
    Plots a bar chart of data nullities by column.

    :param df: The DataFrame whose completeness is being nullity matrix mapped.
    :param log: Whether or not to display a logorithmic plot. Defaults to False (linear).
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default). See
    `nullity_filter()` for more information.
    :param n: The cap on the number of columns to include in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param p: The cap on the percentage fill of the columns in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param sort: The sort to apply to the heatmap. Should be one of "ascending", "descending", or None. See
    `nullity_sort()` for more information.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter. Defaults to (24,
    10).
    :param fontsize: The figure's font size. This default to 16.
    :param labels: Whether or not to display the column names. Would need to be turned off on particularly large
    displays. Defaults to True.
    :param color: The color of the filled columns. Default is a medium dark gray: the RGB multiple `(0.25, 0.25, 0.25)`.
    :return: If `inline` is True, the underlying `matplotlib.figure` object. Else, nothing.
    """
    # Get counts.
    nullity_counts = len(df) - df.isnull().sum()

    # Apply filters and sorts.
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort)

    # Create the basic plot.
    fig = plt.figure(figsize=figsize)
    (nullity_counts / len(df)).plot(kind='bar', figsize=figsize, fontsize=fontsize, color=color, log=log)

    # Get current axis.
    ax1 = plt.gca()

    # Start appending elements, starting with a modified bottom x axis.
    if labels or (labels is None and len(df.columns) <= 50):
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)

        # Create the numerical ticks.
        ax2 = ax1.twinx()
        if not log:
            # Simple if the plot is ordinary.
            ax2.set_yticks(ax1.get_yticks())
            ax2.set_yticklabels([int(n*len(df)) for n in ax1.get_yticks()], fontsize=fontsize)
        else:
            # For some reason when a logarithmic plot is specified `ax1` always contains two more ticks than actually
            # appears in the plot. For example, if we do `msno.histogram(collisions.sample(500), log=True)` the contents
            # of the naive `ax1.get_yticks()` is [1.00000000e-03, 1.00000000e-02, 1.00000000e-01, 1.00000000e+00,
            # 1.00000000e+01]. The fix is to ignore the first and last entries.
            #
            # Also note that when a log scale is used, we have to make it match the `ax1` layout ourselves.
            ax2.set_yscale('log')
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_yticks(ax1.get_yticks()[1:-1])
            ax2.set_yticklabels([int(n*len(df)) for n in ax1.get_yticks()[1:-1]], fontsize=fontsize)

    # Create the third axis, which displays columnar totals above the rest of the plot.
    ax3 = ax1.twiny()
    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticklabels(nullity_counts.values, fontsize=fontsize, rotation=45, ha='left')
    ax3.grid(False)

    # Display.
    if inline:
        plt.show()
    else:
        return fig


def heatmap(df, inline=True,
            filter=None, n=0, p=0, sort=None,
            figsize=(20, 12), fontsize=16, labels=True, cmap='RdBu'
            ):
    """
    Presents a `seaborn` heatmap visualization of nullity correlation in the given DataFrame.
    
    Note that this visualization has no special support for large datasets. For those, try the dendrogram instead.
    

    :param df: The DataFrame whose completeness is being heatmapped.
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default). See
    `nullity_filter()` for more information.
    :param n: The cap on the number of columns to include in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param p: The cap on the percentage fill of the columns in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param sort: The sort to apply to the heatmap. Should be one of "ascending", "descending", or None. See
    `nullity_sort()` for more information.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter which defaults to (20, 12).
    :param fontsize: The figure's font size.
    :param labels: Whether or not to label each matrix entry with its correlation (default is True).
    :param cmap: What `matplotlib` colormap to use. Defaults to `RdBu`.
    :param inline: Whether or not the figure is inline. If it's not then instead of getting plotted, this method will
    return its figure.
    :return: If `inline` is True, the underlying `matplotlib.figure` object. Else, nothing.
    """
    # Apply filters and sorts.
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort)

    # Set up the figure.
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])

    # Pre-processing: remove completely filled or completely empty variables.
    df = df[[i for i, n in enumerate(np.var(df.isnull(), axis='rows')) if n > 0]]

    # Create and mask the correlation matrix.
    corr_mat = df.isnull().corr()
    # corr_mat = corr_mat.replace(np.nan, 1)
    # corr_mat[np.isnan(corr_mat)] = 0
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True

    # Set fontsize.
    # fontsize = _set_font_size(fig, df, fontsize)

    # Construct the base heatmap.
    if labels:
        sns.heatmap(corr_mat, mask=mask, cmap=cmap, ax=ax0, cbar=False,
                    annot=True, annot_kws={"size": fontsize - 2})
    else:
        sns.heatmap(corr_mat, mask=mask, cmap=cmap, ax=ax0, cbar=False)

    # Apply visual corrections and modifications.
    ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=45, ha='left', fontsize=fontsize)
    ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(), fontsize=fontsize, rotation=0)
    ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(), rotation=0, fontsize=fontsize)

    ax0.xaxis.tick_top()
    ax0.patch.set_visible(False)

    # Fix up annotation label rendering.
    for text in ax0.texts:
        t = float(text.get_text())
        if 0.95 <= t < 1:
            text.set_text("<1")
        elif -1 < t <= -0.95:
            text.set_text(">-1")
        elif t == 1:
            text.set_text("1")
        elif t == -1:
            text.set_text("-1")
        elif -0.05 < t < 0.05:
            text.set_text("")
        else:
            text.set_text(round(t, 1))

    if inline:
        plt.show()
    else:
        return fig

    
def dendrogram(df, method='average',
               filter=None, n=0, p=0, sort=None,
               orientation=None, figsize=None,
               fontsize=16, inline=True
               ):
    """
    Fits a `scipy` hierarchical clustering algorithm to the given DataFrame's variables and visualizes the results as
    a `scipy` dendrogram.
    
    The default vertical display will fit up to 50 columns. If more than 50 columns are specified and orientation is
    left unspecified the dendrogram will automatically swap to a horizontal display to fit the additional variables.

    :param df: The DataFrame whose completeness is being dendrogrammed.
    :param method: The distance measure being used for clustering. This is a parameter that is passed to 
    `scipy.hierarchy`.
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default). See
    `nullity_filter()` for more information.
    :param n: The cap on the number of columns to include in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param p: The cap on the percentage fill of the columns in the filtered DataFrame. See  `nullity_filter()` for
    more information.
    :param sort: The sort to apply to the heatmap. Should be one of "ascending", "descending", or None. See
    `nullity_sort()` for more information.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter which defaults to `(25, 10)`.
    :param fontsize: The figure's font size.
    :param orientation: The way the dendrogram is oriented. Defaults to top-down if there are less than or equal to 50
    columns and left-right if there are more.
    :param inline: Whether or not the figure is inline. If it's not then instead of getting plotted, this method will
    return its figure.
    :return: If `inline` is True, the underlying `matplotlib.figure` object. Else, nothing.
    """
    # Figure out the appropriate figsize.
    if not figsize:
        if len(df.columns) <= 50 or orientation == 'top' or orientation == 'bottom':
            figsize = (25, 10)
        else:
            figsize = (25, (25 + len(df.columns) - 50)*0.5)
    
    # Set up the figure.
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])

    # Apply filters and sorts.
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort)

    # Link the hierarchical output matrix.
    x = np.transpose(df.isnull().astype(int).values)
    z = hierarchy.linkage(x, method)

    # Figure out orientation.
    if not orientation:
        if len(df.columns) > 50:
            orientation = 'left'
        else:
            orientation = 'bottom'

    # Construct the base dendrogram.
    ret = hierarchy.dendrogram(z,
                               orientation=orientation,
                               labels=df.columns.tolist(),
                               distance_sort='descending',
                               link_color_func=lambda c: 'black',
                               leaf_font_size=fontsize,
                               ax=ax0
                               )

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(b=False)
    if orientation == 'bottom':
        ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.patch.set_visible(False)

    # Set up the categorical axis labels.
    if orientation == 'bottom':
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=45, ha='left')
    elif orientation == 'top':
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=45, ha='right')
    if orientation == 'bottom' or orientation == 'top':
        ax0.tick_params(axis='y', labelsize=20)
    else:
        ax0.tick_params(axis='x', labelsize=20)

    if inline:
        plt.show()
    else:
        return fig


def _calculate_geographic_nullity(geo_group, x_col, y_col):
    """
    Helper method which calculates the nullity of a DataFrame. Factored out of and used within `geoplot`.
    """
    # Aggregate by point and fetch a list of non-null coordinate pairs, which is returned.
    point_groups = geo_group.groupby([x_col, y_col])
    points = [point for point in point_groups.groups.keys() if pd.notnull(point[0]) and pd.notnull(point[1])]
    # Calculate nullities by location, then take their average within the overall feature.
    counts = np.sum(point_groups.count().values, axis=1)
    entries = point_groups.size()
    width = len(geo_group.columns)
    # Remove empty (NaN, NaN) points.
    if len(entries) > 0:  # explicit check to avoid a Runtime Warning
        geographic_nullity = np.average(1 - counts / width / entries)
        return points, geographic_nullity
    else:
        return points, np.nan


def geoplot(df, x=None, y=None, coordinates=None, by=None, geometry=None, cutoff=None, histogram=False,
            figsize=(25, 10), fontsize=8, inline=True):
    """
    Generates a geographical data nullity heatmap, which shows the distribution of missing data across geographic
    regions. The precise output depends on the inputs provided. In increasing order of usefulness:

    *   If no geographical context is provided, a quadtree is computed and nullities are rendered as abstract
        geopgrahical squares.
    *   If geographical context is provided in the form of a column of geographies (region, borough. ZIP code,
        etc.) in the `DataFrame`, convex hulls are computed for each of the point groups and the heatmap is generated
        within them.
    *   If geographical context is provided *and* a separate geometry is provided, a heatmap is generated for each
        point group within this geograpby instead.

    :param df: The DataFrame whose completeness is being mapped.
    :param x: The x variable: probably a coordinate (longitude), possibly some other floating point value. May be a
    string (pointing to a column of df) or an iterable.
    :param y: The y variable: probably a coordinate (latitude), possibly some other floating point value. May be a
    string (pointing to a column of df) or an iterable.
    :param coordinates: A coordinate tuple iterable, or column thereof in the given DataFrame. One of x AND y OR
    coordinates must be specified, but not both.
    :param by: If you would like to aggregate your geometry by some geospatial attribute of the underlying DataFrame,
    name that column here.
    :param geometry: If you would like to provide your own geometries for your aggregation, instead of relying on
    (functional, but not pretty) convex hulls, provide them here. This parameter is expected to be a dict or Series
    of `shapely.Polygon` or `shapely.MultiPolygon` objects. It's ignored if `by` is not specified.
    :param cutoff: If no aggregation is specified, this parameter sets the minimum number of observations to include in
    each square. If not provided, set to 50 or 5% of the total size of the dataset, whichever is smaller. If `by` is
    specified this parameter is ignored.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter which defaults to (25, 10).
    :param histogram: Whether or not to plot a histogram of data distributions below the map. Defaults to False.
    :param fontsize: If `hist` is specified, this parameter specifies the size of the tick labels. Ignored if `hist`
    is not specified. Defaults to 8.
    :param inline: Whether or not the figure is inline. If it's not then instead of getting plotted, this method will
    return its figure.
    :return: If `inline` is True, the underlying `matplotlib.figure` object. Else, nothing.
    """
    import shapely.geometry
    import descartes
    import matplotlib.cm
    # We produce a coordinate column in-place in a function-local copy of the `DataFrame`.
    # This seems specious, and sort of is, but is necessary because the internal `pandas` aggregation methods
    # (`pd.core.groupby.DataFrameGroupBy.count` specifically) are optimized to run two orders of magnitude faster than
    # user-defined external `groupby` operations. For example:
    # >>> %time df.head(100000).groupby(lambda ind: df.iloc[ind]['LOCATION']).count()
    # Wall time: 12.7 s
    # >>> %time df.head(100000).groupby('LOCATION').count()
    # Wall time: 96 ms
    x_col = '__x'
    y_col = '__y'
    if x and y:
        if isinstance(x, str) and isinstance(y, str):
            x_col = x
            y_col = y
        else:
            df['__x'] = x
            df['__y'] = y
    elif coordinates:
        if isinstance(coordinates, str):
            # Quick conversion to enable fancy numpy indexing. This allows fast operations like `df[coord_col][0,...]`
            coord_col = np.array([np.array(e) if pd.notnull(e) else np.array(np.nan, np.nan) for e in df[coordinates]])
            df['__x'] = coord_col[:, 0]
            df['__y'] = coord_col[:, 1]
        else:
            # cf. Above.
            coord_col = np.array([np.array(e) for e in coordinates])
            df['__x'] = coord_col[:, 0]
            df['__y'] = coord_col[:, 1]
    else:
        raise IndexError("x AND y OR coordinates parameters expected.")

    # Set the cutoff variable.
    if cutoff is None: cutoff = np.min([50, 0.05 * len(df)])

    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    # In case we're given something to categorize by, use that.
    if by:
        nullity_dict = dict()
        # This loop calculates and stores geographic feature averages and their polygons.
        for identifier, geo_group in df.groupby(by):  # ex. ('BRONX', <pd.DataFrame with 10511 objects>)
            # A single observation in the group will produce a `Point` hull, while two observations in the group
            # will produce a `LineString` hull. Neither of these is desired, nor accepted by `PatchCollection`
            # further on. So we remove these cases by filtering them (1) before and (2) after aggregation steps.
            # cf. http://toblerity.org/shapely/manual.html#object.convex_hull
            if not len(geo_group) > 2:
                continue

            # The following subroutine groups `geo_group` by `x_col` and `y_col`, and calculates and returns
            # a list of points in the group (`points`) as well as its overall nullity (`geographic_nullity`).
            points, geographic_nullity = _calculate_geographic_nullity(geo_group, x_col, y_col)

            # If thinning the points, above, reduced us below the threshold for a proper polygonal hull (See the
            # note at the beginning of thos loop), stop here.
            if not len(points) > 2:
                continue

            # If no geometry is provided, calculate and store the hulls and averages.
            if geometry is None:
                    hull = shapely.geometry.MultiPoint(points).convex_hull
                    nullity_dict[identifier] = {'nullity': geographic_nullity, 'shapes': [hull]}

            # If geometry is provided, use that instead.
            else:
                geom = geometry[identifier]
                polygons = []
                # Valid polygons are simple polygons (`shapely.geometry.Polygon`) and complex multi-piece polygons
                # (`shapely.geometry.MultiPolygon`). The latter is an iterable of its components, so if the shape is
                # a `MultiPolygon`, append it as that list. Otherwise if the shape is a basic `Polygon`,
                # append a list with one element, the `Polygon` itself.
                if isinstance(geom, shapely.geometry.MultiPolygon):
                    polygons = shapely.ops.cascaded_union([p for p in geometry])
                else:  # shapely.geometry.Polygon
                    polygons = [geom]
                nullity_dict[identifier] = {'nullity': geographic_nullity, 'shapes': polygons}

        # Prepare a colormap.
        nullities = [nullity_dict[key]['nullity'] for key in nullity_dict.keys()]
        colors = matplotlib.cm.YlOrRd(plt.Normalize(0, 1)(nullities))

        # Now we draw.
        for i, polygons in enumerate([(nullity_dict[key]['shapes']) for key in nullity_dict.keys()]):
            for polygon in polygons:
                ax.add_patch(descartes.PolygonPatch(polygon, fc=colors[i], ec='white', alpha=0.8, zorder=4))
        ax.axis('equal')

        # Remove extraneous plotting elements.
        ax.grid(b=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.patch.set_visible(False)

    # In case we aren't given something to categorize by, we choose a spatial representation that's reasonably
    # efficient and informative: quadtree rectangles.
    # Note: SVD could perhaps be applied to the axes to discover point orientation and realign the grid to match
    # that, but I'm uncertain that this computationally acceptable and, in the case of comparisons, even a good
    # design choice. Perhaps this could be implemented at a later date.
    else:
        df = df[(pd.notnull(df[x_col])) & (pd.notnull(df[y_col]))]
        min_x, max_x = df[x_col].min(), df[x_col].max()
        min_y, max_y = df[y_col].min(), df[y_col].max()

        rectangles = []

        # Recursive quadtree. This subroutine, when, builds a dictionary of squares, stored by tuples keyed with
        # (min_x, max_x, min_y, max_y), whose values are the nullity of squares containing less than 100 observations.
        def squarify(_min_x, _max_x, _min_y, _max_y, df):
            arr = df[[x_col, y_col]].values
            points_inside = df[(_min_x < arr[:,0]) &
                               (arr[:,0] < _max_x) &
                               (_min_y < arr[:,1]) &
                               (arr[:,1] < _max_y)]
            if len(points_inside) < cutoff:
                # The following subroutine groups `geo_group` by `x_col` and `y_col`, and calculates and returns
                # a list of points in the group (`points`) as well as its overall nullity (`geographic_nullity`). The
                # first of these calculations is ignored.
                _, square_nullity = _calculate_geographic_nullity(points_inside, x_col, y_col)
                rectangles.append(((_min_x, _max_x,_min_y, _max_y), square_nullity))
            else:
                _mid_x, _mid_y = (_min_x + _max_x) / 2, (_min_y + _max_y) / 2
                squarify(_min_x, _mid_x, _mid_y, _max_y, points_inside)
                squarify(_min_x, _mid_x, _min_y, _mid_y, points_inside)
                squarify(_mid_x, _max_x, _mid_y, _max_y, points_inside)
                squarify(_mid_x, _max_x, _min_y, _mid_y, points_inside)

        # Populate the `squares` array, per the above.
        squarify(min_x, max_x, min_y, max_y, df)

        # Prepare a colormap.
        # Many of the squares at the bottom of the quadtree will be inputted into the colormap as NaN values,
        # which matplotlib will map over as minimal values. We of course don't want that, so we pull the bottom out
        # of it.
        nullities = [nullity for _, nullity in rectangles]
        cmap = matplotlib.cm.YlOrRd
        colors = [cmap(n) if pd.notnull(n) else [1,1,1,1]
                  for n in plt.Normalize(0, 1)(nullities)]

        # Now we draw.
        for i, ((min_x, max_x, min_y, max_y), _) in enumerate(rectangles):
            square = shapely.geometry.Polygon([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
            ax.add_patch(descartes.PolygonPatch(square, fc=colors[i], ec='white', alpha=1, zorder=4))
        ax.axis('equal')

        # Remove extraneous plotting elements.
        ax.grid(b=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.patch.set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.patch.set_visible(False)

    if histogram:
        # Add a histogram.
        sns.set_style(None)
        nonnan_nullities = [n for n in nullities if pd.notnull(n)]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="25%", pad=0.00)
        sns.distplot(pd.Series(nonnan_nullities), ax=cax, color=list(np.average(colors, axis=0)))

        cax.grid(b=False)
        # cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_visible(False)
        cax.xaxis.set_ticks_position('none')
        cax.yaxis.set_ticks_position('none')
        cax.spines['top'].set_visible(False)
        cax.spines['right'].set_visible(False)
        cax.spines['bottom'].set_visible(False)
        cax.spines['left'].set_visible(False)
        cax.patch.set_visible(False)
        cax.tick_params(labelsize=fontsize)

    # Display.
    # Display.
    if inline:
        plt.show()
    else:
        return fig

