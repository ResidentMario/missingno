import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns
import pandas as pd
from .utils import nullity_filter, nullity_sort
import warnings


def matrix(
    df, filter=None, n=0, p=0, sort=None, figsize=(25, 10), width_ratios=(15, 1),
    color=(0.25, 0.25, 0.25), fontsize=16, labels=None, label_rotation=45, sparkline=True,
    freq=None, ax=None
):
    """
    A matrix visualization of the nullity of the given DataFrame.

    :param df: The `DataFrame` being mapped.
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
    :param n: The max number of columns to include in the filtered DataFrame.
    :param p: The max percentage fill of the columns in the filtered DataFrame.
    :param sort: The row sort order to apply. Can be "ascending", "descending", or None.
    :param figsize: The size of the figure to display.
    :param fontsize: The figure's font size. Default to 16.
    :param labels: Whether or not to display the column names. Defaults to the underlying data labels when there are
        50 columns or less, and no labels when there are more than 50 columns.
    :param label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
    :param sparkline: Whether or not to display the sparkline. Defaults to True.
    :param width_ratios: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15, 1)`.
        Does nothing if `sparkline=False`.
    :param color: The color of the filled columns. Default is `(0.25, 0.25, 0.25)`.
    :return: The plot axis.
    """
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort, axis='columns')

    height = df.shape[0]
    width = df.shape[1]

    # z is the color-mask array, g is a NxNx3 matrix. Apply the z color-mask to set the RGB of each pixel.
    z = df.notnull().values
    g = np.zeros((height, width, 3), dtype=np.float32)

    g[z < 0.5] = [1, 1, 1]
    g[z > 0.5] = color

    # Set up the matplotlib grid layout. A unary subplot if no sparkline, a left-right splot if yes sparkline.
    if ax is None:
        plt.figure(figsize=figsize)
        if sparkline:
            gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
            gs.update(wspace=0.08)
            ax1 = plt.subplot(gs[1])
        else:
            gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0])
    else:
        if sparkline is not False:
            warnings.warn(
                "Plotting a sparkline on an existing axis is not currently supported. "
                "To remove this warning, set sparkline=False."
            )
            sparkline = False
        ax0 = ax

    # Create the nullity plot.
    ax0.imshow(g, interpolation='none')

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(visible=False)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)

    # Set up and rotate the column ticks. The labels argument is set to None by default. If the user specifies it in
    # the argument, respect that specification. Otherwise display for <= 50 columns and do not display for > 50.
    if labels or (labels is None and len(df.columns) <= 50):
        ha = 'left'
        ax0.set_xticks(list(range(0, width)))
        ax0.set_xticklabels(list(df.columns), rotation=label_rotation, ha=ha, fontsize=fontsize)
    else:
        ax0.set_xticks([])

    # Adds Timestamps ticks if freq is not None, else set up the two top-bottom row ticks.
    if freq:
        ts_list = []

        if type(df.index) == pd.PeriodIndex:
            ts_array = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))

        elif type(df.index) == pd.DatetimeIndex:
            ts_array = pd.date_range(df.index[0], df.index[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index[0], df.index[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))
        else:
            raise KeyError('Dataframe index must be PeriodIndex or DatetimeIndex.')
        try:
            for value in ts_array:
                ts_list.append(df.index.get_loc(value))
        except KeyError:
            raise KeyError('Could not divide time index into desired frequency.')

        ax0.set_yticks(ts_list)
        ax0.set_yticklabels(ts_ticks, fontsize=int(fontsize / 16 * 20), rotation=0)
    else:
        ax0.set_yticks([0, df.shape[0] - 1])
        ax0.set_yticklabels([1, df.shape[0]], fontsize=int(fontsize / 16 * 20), rotation=0)

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

        # Set up the sparkline, remove the border element.
        ax1.grid(visible=False)
        ax1.set_aspect('auto')
        # GH 25
        if int(mpl.__version__[0]) <= 1:
            ax1.set_axis_bgcolor((1, 1, 1))
        else:
            ax1.set_facecolor((1, 1, 1))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_ymargin(0)

        # Plot sparkline---plot is sideways so the x and y axis are reversed.
        ax1.plot(y_range, x_domain, color=color)

        if labels:
            # Figure out what case to display the label in: mixed, upper, lower.
            label = 'Data Completeness'
            if str(df.columns[0]).islower():
                label = label.lower()
            if str(df.columns[0]).isupper():
                label = label.upper()

            # Set up and rotate the sparkline label.
            ha = 'left'
            ax1.set_xticks([min_completeness + (max_completeness - min_completeness) / 2])
            ax1.set_xticklabels([label], rotation=label_rotation, ha=ha, fontsize=fontsize)
            ax1.xaxis.tick_top()
            ax1.set_yticks([])
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])

        # Add maximum and minimum labels, circles.
        ax1.annotate(max_completeness,
                     xy=(max_completeness, max_completeness_index),
                     xytext=(max_completeness + 2, max_completeness_index),
                     fontsize=int(fontsize / 16 * 14),
                     va='center',
                     ha='left')
        ax1.annotate(min_completeness,
                     xy=(min_completeness, min_completeness_index),
                     xytext=(min_completeness - 2, min_completeness_index),
                     fontsize=int(fontsize / 16 * 14),
                     va='center',
                     ha='right')

        ax1.set_xlim([min_completeness - 2, max_completeness + 2])  # Otherwise the circles are cut off.
        ax1.plot([min_completeness], [min_completeness_index], '.', color=color, markersize=10.0)
        ax1.plot([max_completeness], [max_completeness_index], '.', color=color, markersize=10.0)

        # Remove tick mark (only works after plotting).
        ax1.xaxis.set_ticks_position('none')

    return ax0


def bar(
    df, figsize=None, fontsize=16, labels=None, label_rotation=45, log=False, color='dimgray',
    filter=None, n=0, p=0, sort=None, ax=None, orientation=None
):
    """
    A bar chart visualization of the nullity of the given DataFrame.

    :param df: The input DataFrame.
    :param log: Whether or not to display a logarithmic plot. Defaults to False (linear).
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
    :param n: The cap on the number of columns to include in the filtered DataFrame.
    :param p: The cap on the percentage fill of the columns in the filtered DataFrame.
    :param sort: The column sort order to apply. Can be "ascending", "descending", or None.
    :param figsize: The size of the figure to display.
    :param fontsize: The figure's font size. This default to 16.
    :param labels: Whether or not to display the column names. Would need to be turned off on particularly large
        displays. Defaults to True.
    :param label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
    :param color: The color of the filled columns. Default to the RGB multiple `(0.25, 0.25, 0.25)`.
    :param orientation: The way the bar plot is oriented. Defaults to vertical if there are less than or equal to 50
        columns and horizontal if there are more.
    :return: The plot axis.
    """
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort, axis='rows')
    nullity_counts = len(df) - df.isnull().sum()

    if orientation is None:
        if len(df.columns) > 50:
            orientation = 'left'
        else:
            orientation = 'bottom'

    if ax is None:
        ax1 = plt.gca()
        if figsize is None:
            if len(df.columns) <= 50 or orientation == 'top' or orientation == 'bottom':
                figsize = (25, 10)
            else:
                figsize = (25, (25 + len(df.columns) - 50) * 0.5)
    else:
        ax1 = ax
        figsize = None  # for behavioral consistency with other plot types, re-use the given size

    plot_args = {'figsize': figsize, 'fontsize': fontsize, 'log': log, 'color': color, 'ax': ax1}
    if orientation == 'bottom':
        (nullity_counts / len(df)).plot.bar(**plot_args)
    else:
        (nullity_counts / len(df)).plot.barh(**plot_args)

    axes = [ax1]

    # Start appending elements, starting with a modified bottom x axis.
    if labels or (labels is None and len(df.columns) <= 50):
        ax1.set_xticklabels(
            ax1.get_xticklabels(), rotation=label_rotation, ha='right', fontsize=fontsize
        )

        # Create the numerical ticks.
        ax2 = ax1.twinx()
        axes.append(ax2)
        if not log:
            ax1.set_ylim([0, 1])
            ax2.set_yticks(ax1.get_yticks())
            ax2.set_yticklabels([int(n * len(df)) for n in ax1.get_yticks()], fontsize=fontsize)
        else:
            # For some reason when a logarithmic plot is specified `ax1` always contains two more ticks than actually
            # appears in the plot. The fix is to ignore the first and last entries. Also note that when a log scale
            # is used, we have to make it match the `ax1` layout ourselves.
            ax2.set_yscale('log')
            ax2.set_ylim(ax1.get_ylim())
        ax2.set_yticklabels([int(n * len(df)) for n in ax1.get_yticks()], fontsize=fontsize)

        # Create the third axis, which displays columnar totals above the rest of the plot.
        ax3 = ax1.twiny()
        axes.append(ax3)
        ax3.set_xticks(ax1.get_xticks())
        ax3.set_xlim(ax1.get_xlim())
        ax3.set_xticklabels(
            nullity_counts.values, fontsize=fontsize, rotation=label_rotation, ha='left'
        )
    else:
        # Create the numerical ticks.
        ax2 = ax1.twinx()

        axes.append(ax2)
        if not log:
            # Width
            ax1.set_xlim([0, 1])

            # Bottom
            ax2.set_xticks(ax1.get_xticks())
            ax2.set_xticklabels([int(n * len(df)) for n in ax1.get_xticks()], fontsize=fontsize)

            # Right
            ax2.set_yticks(ax1.get_yticks())
            ax2.set_yticklabels(nullity_counts.values, fontsize=fontsize, ha='left')
        else:
            # For some reason when a logarithmic plot is specified `ax1` always contains two more ticks than actually
            # appears in the plot. The fix is to ignore the first and last entries. Also note that when a log scale
            # is used, we have to make it match the `ax1` layout ourselves.
            ax1.set_xscale('log')
            ax1.set_xlim(ax1.get_xlim())

            # Bottom
            ax2.set_xticks(ax1.get_xticks())
            ax2.set_xticklabels([int(n * len(df)) for n in ax1.get_xticks()], fontsize=fontsize)

            # Right
            ax2.set_yticks(ax1.get_yticks())
            ax2.set_yticklabels(nullity_counts.values, fontsize=fontsize, ha='left')

        # Create the third axis, which displays columnar totals above the rest of the plot.
        ax3 = ax1.twiny()

        axes.append(ax3)
        ax3.set_yticks(ax1.get_yticks())
        if log:
            ax3.set_xscale('log')
            ax3.set_xlim(ax1.get_xlim())
        ax3.set_ylim(ax1.get_ylim())

    ax3.grid(False)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

    return ax1


def heatmap(
    df, filter=None, n=0, p=0, sort=None, figsize=(20, 12), fontsize=16, labels=True,
    label_rotation=45, cmap='RdBu', vmin=-1, vmax=1, cbar=True, ax=None
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
    :param sort: The column sort order to apply. Can be "ascending", "descending", or None.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter which defaults to (20, 12).
    :param fontsize: The figure's font size.
    :param labels: Whether or not to label each matrix entry with its correlation (default is True).
    :param label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
    :param cmap: What `matplotlib` colormap to use. Defaults to `RdBu`.
    :param vmin: The normalized colormap threshold. Defaults to -1, e.g. the bottom of the color scale.
    :param vmax: The normalized colormap threshold. Defaults to 1, e.g. the bottom of the color scale.
    :return: The plot axis.
    """
    # Apply filters and sorts, set up the figure.
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort, axis='rows')

    if ax is None:
        plt.figure(figsize=figsize)
        ax0 = plt.gca()
    else:
        ax0 = ax

    # Remove completely filled or completely empty variables.
    df = df.iloc[:, [i for i, n in enumerate(np.var(df.isnull(), axis='rows')) if n > 0]]

    # Create and mask the correlation matrix. Construct the base heatmap.
    corr_mat = df.isnull().corr()
    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True

    if labels:
        sns.heatmap(corr_mat, mask=mask, cmap=cmap, ax=ax0, cbar=cbar,
                    annot=True, annot_kws={'size': fontsize - 2},
                    vmin=vmin, vmax=vmax)
    else:
        sns.heatmap(corr_mat, mask=mask, cmap=cmap, ax=ax0, cbar=cbar,
                    vmin=vmin, vmax=vmax)

    # Apply visual corrections and modifications.
    ax0.xaxis.tick_bottom()
    ax0.set_xticklabels(
        ax0.xaxis.get_majorticklabels(), rotation=label_rotation, ha='right', fontsize=fontsize
    )
    ax0.set_yticklabels(ax0.yaxis.get_majorticklabels(), rotation=0, fontsize=fontsize)
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.patch.set_visible(False)

    for text in ax0.texts:
        t = float(text.get_text())
        if 0.95 <= t < 1:
            text.set_text('<1')
        elif -1 < t <= -0.95:
            text.set_text('>-1')
        elif t == 1:
            text.set_text('1')
        elif t == -1:
            text.set_text('-1')
        elif -0.05 < t < 0.05:
            text.set_text('')
        else:
            text.set_text(round(t, 1))

    return ax0


def dendrogram(
    df, method='average', filter=None, n=0, p=0, orientation=None, figsize=None, fontsize=16,
    label_rotation=45, ax=None
):
    """
    Fits a `scipy` hierarchical clustering algorithm to the given DataFrame's variables and visualizes the results as
    a `scipy` dendrogram.

    The default vertical display will fit up to 50 columns. If more than 50 columns are specified and orientation is
    left unspecified the dendrogram will automatically swap to a horizontal display to fit the additional variables.

    :param df: The DataFrame whose completeness is being dendrogrammed.
    :param method: The distance measure being used for clustering. This is a parameter that is passed to
        `scipy.hierarchy`.
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
    :param n: The cap on the number of columns to include in the filtered DataFrame.
    :param p: The cap on the percentage fill of the columns in the filtered DataFrame.
    :param figsize: The size of the figure to display. This is a `matplotlib` parameter which defaults to `(25, 10)`.
    :param fontsize: The figure's font size.
    :param orientation: The way the dendrogram is oriented. Defaults to top-down if there are less than or equal to 50
        columns and left-right if there are more.
    :param label_rotation: What angle to rotate the text labels to. Defaults to 45 degrees.
    :return: The plot axis.
    """
    if not figsize:
        if len(df.columns) <= 50 or orientation == 'top' or orientation == 'bottom':
            figsize = (25, 10)
        else:
            figsize = (25, (25 + len(df.columns) - 50) * 0.5)

    if ax is None:
        plt.figure(figsize=figsize)
        ax0 = plt.gca()
    else:
        ax0 = ax

    df = nullity_filter(df, filter=filter, n=n, p=p)

    # Link the hierarchical output matrix, figure out orientation, construct base dendrogram.
    x = np.transpose(df.isnull().astype(int).values)
    z = hierarchy.linkage(x, method)

    if not orientation:
        if len(df.columns) > 50:
            orientation = 'left'
        else:
            orientation = 'bottom'

    hierarchy.dendrogram(
        z,
        orientation=orientation,
        labels=df.columns.tolist(),
        distance_sort='descending',
        link_color_func=lambda c: 'black',
        leaf_font_size=fontsize,
        ax=ax0
    )

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(visible=False)
    if orientation == 'bottom':
        ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.patch.set_visible(False)

    # Set up the categorical axis labels and draw.
    if orientation == 'bottom':
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=label_rotation, ha='left')
    elif orientation == 'top':
        ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=label_rotation, ha='right')
    if orientation == 'bottom' or orientation == 'top':
        ax0.tick_params(axis='y', labelsize=int(fontsize / 16 * 20))
    else:
        ax0.tick_params(axis='x', labelsize=int(fontsize / 16 * 20))

    return ax0
