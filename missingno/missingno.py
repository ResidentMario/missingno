import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns
import pandas as pd


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


# def _set_font_size(fig, df, fontsize):
#     """
#     Guesses an appropriate fontsize for the given columnar visualization text labels.
#     Used if a fontsize is not provided via a parameter.
#     """
#     if fontsize:
#         return fontsize
#     else:
#         return max(min(20, int((fig.get_size_inches()[1] * 0.7) * fig.dpi / len(df.columns))), 16)


def matrix(df,
           filter=None, n=0, p=0, sort=None,
           figsize=(25, 10), width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
           fontsize=16, labels=None, sparkline=True, inline=True, flip=None
           ):
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
    :param flip: The default matrix orientation is top-down, with column on the vertical and rows on the horizontal---
    just like a table. However, for large displays (> 50 by default) display in this format becomes uncomfortable, so
    the display gets flipped. This parameter is specified to be True if there are more than 50 columns and False
    otherwise.
    :return: Returns the underlying `matplotlib.figure` object.
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

    # Set fontsize.
    # fontsize = _set_font_size(fig, df, fontsize)

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
    if labels or (labels == None and len(df.columns) <= 50):
        ha = 'left'
        ax0.set_xticks(list(range(0, width)))
        ax0.set_xticklabels(list(df.columns), rotation=45, ha=ha, fontsize=fontsize)
    else:
        ax0.set_xticks([])

    # Set up the two top-bottom row ticks.
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
        ax1.set_axis_bgcolor((1, 1, 1))
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
    :return: Returns the underlying `matplotlib.figure` object.
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
    :return: Returns the underlying `matplotlib.figure` object.
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

    # Set font size.
    # if orientation == 'top' or orientation == 'bottom':
    #     fontsize = _set_font_size(fig, df, fontsize)
    # else:
    #     fontsize = 20

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


def geoplot(df, x=None, y=None, coordinates=None, by=None):
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
            # df['__coords'] = list(zip(df[x], df[y]))
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
    # In case we're given something to categorize by, use that.
    if by:
        # This loop calculates and stores geographic feature averages and their convex hulls.
        nullities = dict()
        for identifier, geo_group in df.groupby(by):  # ex. ('BRONX', <pd.DataFrame with 10511 objects>)
            # A single observation in the group will produce a `Point` hull, while two observations in the group
            # will produce a `LineString` hull. Neither of these is desired, nor accepted by `PatchCollection`
            # further on. So we remove these cases.
            # cf. http://toblerity.org/shapely/manual.html#object.convex_hull
            if len(geo_group) > 2:
                point_groups = geo_group.groupby([x_col, y_col])
                # Calculate nullities by location, then take their average within the overall feature.
                counts = point_groups.count().apply(lambda srs: srs.sum(), axis='columns')
                entries = point_groups.apply(len)
                width = len(geo_group.columns)
                if len(entries) > 0:  # explicit check to avoid a Runtime Warning
                    geographic_nullity = np.average(1 - counts / width / entries)
                else:
                    geographic_nullity = np.nan
                # Remove empty (NaN, NaN) points.
                points = [point for point in counts.index.values if pd.notnull(point[0]) and pd.notnull(point[1])]
                # Calculate and store the hulls and averages.
                # If thinning the points, above, reduced us below the threshold for a proper polygonal hull (See the
                # note at the beginning of thos loop), stop here.
                if len(points) > 2:
                    hull = shapely.geometry.MultiPoint(points).convex_hull
                    nullities[identifier] = {'nullity': geographic_nullity, 'shape': hull}
        # Prepare a colormap.
        nullity_avgs = [nullities[key]['nullity'] for key in nullities.keys()]
        cmap = matplotlib.cm.RdBu(plt.Normalize(0, 1)(nullity_avgs))
        # Now we draw.
        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(111)
        for i, hull in enumerate([(nullities[key]['shape']) for key in nullities.keys()]):
            ax.add_patch(descartes.PolygonPatch(hull, fc=cmap[i], ec='white', alpha=0.8, zorder=4))
        ax.axis('image')
        # Remove extraneous plotting elements.
        ax.grid(b=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.patch.set_visible(False)
        plt.show()
        # TODO: Figure out why mplleaflet doesn't integrate with this.
        # TODO: Add color key.
        # TODO: Add geometries option.
    # In case we aren't given something to categorize by, we choose a spatial representation that's reasonably
    # efficient and informative: nested squares.
    # Note: SVD could perhaps be applied to the axes to discover point orientation and realign the grid to match
    # that, but I'm uncertain that this computationally acceptable and, in the case of comparisons, even a good
    # design choice. Perhaps this could be implemented at a later date.
    else:
        df = df[(pd.notnull(df[x_col])) & (pd.notnull(df[y_col]))]
        min_x, max_x = df[x_col].min(), df[x_col].max()
        min_y, max_y = df[y_col].min(), df[y_col].max()
        squares = dict()

        # Recursive quadtree.
        def squarify(_min_x, _max_x, _min_y, _max_y, position):
            points_inside = df[df[[x_col, y_col]]\
                .apply(lambda srs: (_min_x < srs[x_col] < _max_x) and (_min_y < srs[y_col] < _max_y), axis='columns')]
            if len(points_inside) < 100:
                squares[tuple(position)] = len(points_inside)  # temp
                # TODO: Next---make this a nullity count, not a temp point count.
            else:
                mid_x, mid_y = (_min_x + _max_x) / 2, (_min_y + _max_y) / 2
                squarify(_min_x, mid_x, mid_y, _max_y, position + [0])
                squarify(_min_x, mid_x, _min_y, mid_y, position + [1])
                squarify(mid_x, _max_x, mid_y, _max_y, position + [2])
                squarify(mid_x, _max_x, _min_y, mid_y, position + [3])

        squarify(min_x, max_x, min_y, max_y, [0])
        print(squares)