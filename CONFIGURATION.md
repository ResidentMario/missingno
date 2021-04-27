# Advanced Configuration

## Sorting and filtering

`missingno` also provides utility functions for filtering records in your dataset based on completion. These are
useful in particular for filtering through and drilling down into particularly large datasets whose data nullity
issues might otherwise be very hard to visualize or understand.

Let's first apply a `nullity_filter()` to the data. The `filter` parameter controls which result set we
want: either `filter=top` or `filter=bottom`. The `n` parameter controls the maximum number of columns that you want:
 so for example `n=5` makes sure we get *at most* five results. Finally, `p` controls the percentage cutoff. If
 `filter=bottom`, then `p=0.9`  makes sure that our columns are *at most*  90% complete; if `filter=top` we get
 columns which are *at least* 90% complete.

For example, the following query filtered down to only at most 15 columns which are not completely filled.

    >>> filtered_data = msno.nullity_filter(data, filter='bottom', n=15, p=0.999) # or filter='top'
    >>> msno.matrix(filtered_data.sample(250))

![alt text][matrix_filtered]

[matrix_filtered]: http://i.imgur.com/UF6hmL8.png

`nullity_sort()` simply reshuffles your rows by completeness, in either `ascending` or `descending` order. Since it
doesn't affect the underlying data it's mainly useful for `matrix` visualization:


    >>> sorted_data = msno.nullity_sort(data, sort='descending') # or sort='ascending'
    >>> msno.matrix(sorted_data.sample(250))

![alt text][matrix_sorted]

[matrix_sorted]: http://i.imgur.com/qL6zNQj.png

These methods work inline within the visualization methods themselves. For instance, the following is perfectly valid:

    >>> msno.matrix(data.sample(250), filter='top', n=5, p=0.9, sort='ascending')

## Visual configuration
### Lesser parameters

Each of the visualizations provides a further set of lesser configuration parameters for visually tweaking the display.

`matrix`, `bar`, `heatmap`, `dendrogram`, and `geoplot` all provide:

* `figsize`: The size of the figure to display. This is a `matplotlib` parameter which defaults to `(20, 12)`, except
 for large `dendrogram` visualizations, which compute a height on the fly based on the number of variables to display.
* `fontsize`: The figure's font size. The default is `16`.
* `labels`: Whether or not to display the column names. For `matrix` this defaults to `True` for `<=50` variables and
 `False` for `>50`. It always defaults to `True` for `dendrogram` and `heatmap`.
* `inline`: Defaults to `True`, in which case the chart is plotted and nothing is returned. If this is set to `False`
the methods omit plotting and return their visualizations instead.

`matrix` also provides:
* `sparkline`: Set this to `False` to not draw the sparkline.
* `freq`: If you are working with timeseries data (a `pandas` `DataFrame` with a `PeriodIndex` or `DatetimeIndex`)
you can specify and display a [choice of offset](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases).
* `width_ratios`: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15,
    1)`. Does nothing if `sparkline=False`.
* `color`: The color of the filled columns. Defaults to `(0.25, 0.25, 0.25)`.

`bar` also provides:
* `log`: Set this to `True` to use a logarithmic scale.
* `color`: The color of the filled columns. Defaults to `(0.25, 0.25, 0.25)`.


`heatmap` also provides:
* `cmap`: What `matplotlib` [colormap](http://matplotlib.org/users/colormaps.html) to use. Defaults to `RdBu`.


`dendrogram` also provides:
* `orientation`: The [orientation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram)
of the dendrogram. Defaults to `top` if `<=50` columns and
`left` if there are more.
* `method`: The [linkage method](http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage) `scipy.hierarchy` uses for clustering.
`average` is the default argument.

`geoplot` also provides:
* `x` AND `y` OR `coordinates`: A column of points (in either two columns or one) to plot. These are required.
* `by`: A column of values to group points by.
* `geometry`: A hash table (`dict` or `pd.Series` generally) geometries of the groups being aggregated, if available.
* `cutoff`: The minimum number of observations per rectangle in the quadtree display. No effect if a different
display is used. Defaults to `min([50, 0.05*len(df)])`.
* `histogram`: Whether or not to plot the histogram. Defaults to `False`.

### Manipulation with matplotlib
If you are not satisfied with these admittedly basic configuration parameters, the display can be further manipulated
in any way you like using `matplotlib` post-facto.

The best way to do this is to specify `inline=False`, which will cause `missingno` to return the underlying
`matplotlib.axis.Axis` object of the main plot (e.g. only the matrix is returned when plotting the matrix with the sparkline). Anyone with sufficient knowledge of `matplotlib` operations and [the missingno source code](https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py)
can then tweak the display to their liking. For example, the following code will bump the size of the dendrogram
visualization's y-axis labels up from `20` to `30`:

    >>> mat = msno.dendrogram(collisions, inline=False)
    >>> mat.axes[0].tick_params(axis='y', labelsize=30)
