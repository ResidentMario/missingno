# missingno [![PyPi version](https://img.shields.io/pypi/v/missingno.svg)](https://pypi.python.org/pypi/missingno/) ![t](https://img.shields.io/badge/status-stable-green.svg)

Messy datasets? Missing values? `missingno` provides a small toolset of flexible and easy-to-use missing data
visualizations and utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of
your dataset. It's built using `matplotlib`, so it's fast, and takes any `pandas` `DataFrame` input that you throw at
it, so it's flexible. Just `pip install missingno` to get started.

## Quickstart

Examples use the [NYPD Motor Vehicle Collisions Dataset](https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95)
([cleaned up](https://github.com/ResidentMario/motor-vehicle-collisions/blob/master/NYPD%20Motor%20Vehicle%20Collisions.ipynb))
and the [PLUTO Housing Sales Dataset](http://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page)
 ([cleaned up](https://github.com/ResidentMario/nyc-buildings/blob/master/nyc_building_sales.csv)).


In the following walkthrough I take **nullity** to mean whether a particular variable is filled in or not.

### Matrix

The `msno.matrix` nullity matrix is a data-dense display which lets you quickly visually pick out patterns in
 data completion.

    >>> import missingno as msno
    >>> %matplotlib inline
    >>> msno.matrix(collisions.sample(250))

![alt text][two_hundred_fifty]

[two_hundred_fifty]: http://i.imgur.com/DdepYwr.png

At a glance, date, time, the distribution of injuries, and the contribution factor of the first vehicle appear to be
completely populated, while geographic information seems mostly complete, but spottier.

The sparkline at right summarizes the general shape of the data completeness and points out the maximum and minimum
rows.

This visualization will comfortably accommodate up to 50 labelled variables. Past that range labels begin to overlap
or become unreadable, and by default large displays omit them.

    >>> msno.matrix(housing.sample(250))

![alt text][large_matrix]

[large_matrix]: http://i.imgur.com/yITFVju.png

If you are working with time-series data, you can [specify a periodicity](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases)
using the `freq` keyword parameter:

    >>> null_pattern = (np.random.random(1000).reshape((50, 20)) > 0.5).astype(bool)
    >>> null_pattern = pd.DataFrame(null_pattern).replace({False: None})
    >>> msno.matrix(null_pattern.set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M')) , freq='BQ')

![alt text][ts_matrix]

[ts_matrix]: https://cloud.githubusercontent.com/assets/20803912/19692749/470137bc-9a96-11e6-8708-e4d70b91c597.png

<!--
You can override this behavior by specifying `labels=True`. In that case you will also want to set your own
`fontsize` value. These optional parameters are among those covered in more detail in the
[Visual configuration](#visual-configuration) section.
-->

### Bar Chart

`msno.bar` is a simple visualization of nullity by column:

    >>> msno.bar(collisions.sample(500))

![alt text][bar]

[bar]: http://i.imgur.com/lOTN3tm.png

You can switch to a logarithmic scale by specifying `log=True`:

![alt text][bar2]

[bar2]: http://i.imgur.com/YZDaAV3.png

`bar` provides the same information as `matrix`, but in a simpler format.

### Heatmap

The `missingno` correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another:

    >>> msno.heatmap(collisions)

![alt text][heatmap]

[heatmap]: http://i.imgur.com/ESsZRlY.png

In this example, it seems that reports which are filed with an `OFF STREET NAME` variable are less likely to have complete
geographic data.

Nullity correlation ranges from `-1` (if one variable appears the other definitely does not) to `0` (variables appearing
or not appearing have no effect on one another) to `1` (if one variable appears the other definitely also does).

Variables that are always full or always empty have no meaningful correlation, and so are silently removed from the visualization&mdash;in this case for instance the datetime and injury number columns, which are completely filled, are not included.

Entries marked `<1` or `>-1` are have a correlation that is close to being exactingly negative or positive, but is
still not quite perfectly so. This points to a small number of records in the dataset which are erroneous. For
example, in this dataset the correlation between `VEHICLE CODE TYPE 3` and `CONTRIBUTING FACTOR VEHICLE 3` is `<1`,
indicating that, contrary to our expectation, there are a few records which have one or the other, but not both.
These cases will require special attention.

The heatmap works great for picking out data completeness relationships between variable pairs, but its explanatory power
is limited when it comes to larger relationships and it has no particular support for extremely large datasets.


### Dendrogram

The dendrogram allows you to more fully correlate variable completion, revealing trends deeper than the pairwise
ones visible in the correlation heatmap:

    >>> msno.dendrogram(collisions)

![alt text][dendrogram]

[dendrogram]: http://i.imgur.com/6ZBC4af.png

The dendrogram uses a [hierarchical clustering algorithm](http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
(courtesy of `scipy`) to bin variables against one another by their nullity correlation (measured in terms of
binary distance). At each step of the tree the variables are split up based on which combination minimizes the
distance of the remaining clusters. The more monotone the set of variables, the closer their total distance is to
zero, and the closer their average distance (the y-axis) is to zero.

To interpret this graph, read it from a top-down perspective. Cluster leaves which linked together at a distance of
zero fully predict one another's presence&mdash;one variable might always be empty when another is filled, or they
might always both be filled or both empty, and so on. In this specific example the dendrogram glues together the
variables which are required and therefore present in every record.

Cluster leaves which split close to zero, but not at it, predict one another very well, but still imperfectly. If
your own interpretation of the dataset is that these columns actually *are* or *ought to be* match each other in
nullity (for example, as `CONTRIBUTING FACTOR VEHICLE 2` and `VEHICLE TYPE CODE 2` ought to), then the height of the
cluster leaf tells you, in absolute terms, how often the records are "mismatched" or incorrectly filed&mdash;that is,
 how many values you would have to fill in or drop, if you are so inclined.

As with `matrix`, only up to 50 labeled columns will comfortably display in this configuration. However the
`dendrogram` more elegantly handles extremely large datasets by simply flipping to a horizontal configuration.

    >>> msno.dendrogram(housing)

![alt text][large-dendrogram]

[large-dendrogram]: http://i.imgur.com/HDa06O9.png


### Geoplot

One kind of pattern that's particularly difficult to check, where it appears, is geographic distribution. The geoplot
 makes this easy:

    >>> msno.geoplot(collisions.sample(100000), x='LONGITUDE', y='LATITUDE')

![alt-text][large-geoplot]

[large-geoplot]: http://i.imgur.com/4dtGhig.png

If no geographical context can be provided, `geoplot` can be used to compute a
[quadtree](https://en.wikipedia.org/wiki/Quadtree) nullity distribution, as above, which splits the dataset into
statistically significant chunks and colorizes them based on the average nullity of data points within them. In this
case (fortunately for analysis, but unfortunately for the purposes of demonstration) it appears that our dataset's
data nullity is unaffected by geography.

A quadtree analysis works remarkably well in most cases, but will not always be what you want. If you can specify a
geographic grouping within the dataset (using the `by` keyword argument), you can plot your data as a set of
minimum-enclosure [convex hulls](https://en.wikipedia.org/wiki/Convex_hull) instead (the following example also
demonstrates adding a histogram to the display, using the `histogram=True` argument):

    >>> msno.geoplot(collisions.sample(100000), x='LONGITUDE', y='LATITUDE', by='ZIP CODE', histogram=True)

![alt-text][hull-geoplot]

[hull-geoplot]: http://i.imgur.com/3kfKMJO.png

Finally, if you have the *actual* geometries of your grouping (in the form of a `dict` or `pandas` `Series` of
`shapely.Geometry` or `shapely.MultiPolygon` objects), you can dispense with all of this approximation and just plot
*exactly* what you mean:

    >>> msno.geoplot(collisions.sample(1000), x='LONGITUDE', y='LATITUDE', by='BOROUGH', geometry=geom)

![alt-text][true-geoplot]

[true-geoplot]: http://i.imgur.com/fAyxqnk.png

In this case this is the least interesting result of all.

Two technical notes:
* For the geographically inclined, this a [plat carre](https://en.wikipedia.org/wiki/Equirectangular_projection)
projection&mdash;that is, none at all. Not pretty, but functional.
* `geoplot` requires the [`shapely`](https://github.com/Toblerity/Shapely) and [`descartes`](https://pypi.python.org/pypi/descartes) libraries, which are
ancillary to the rest of this package and are thus optional dependencies.

That concludes our tour of `missingno`!

For further details take a look at [this blog post](http://www.residentmar.io/2016/06/12/null-and-missing-data-python.html).

## Other

### Sorting and filtering

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

### Visual configuration
#### Lesser parameters

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

#### Advanced configuration
If you are not satisfied with these admittedly basic configuration parameters, the display can be further manipulated
in any way you like using `matplotlib` post-facto.

The best way to do this is to specify `inline=False`, which will cause `missingno` to return the underlying
`matplotlib.figure.Figure` object. Anyone with sufficient knowledge of `matplotlib` operations and [the missingno source code](https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py)
can then tweak the display to their liking. For example, the following code will bump the size of the dendrogram
visualization's y-axis labels up from `20` to `30`:

    >>> mat = msno.dendrogram(collisions, inline=False)
    >>> mat.axes[0].tick_params(axis='y', labelsize=30)

<!--
Note that if you are running `matplotlib` line in [inline plotting mode](http://www.scipy-lecture.org/intro/matplotlib/matplotlib.html#ipython-and-the-matplotlib-mode)
 (as was done above) it will always plot at the end of the cell anyway, so if you do not want to plot the same
 visualization multiple times you will want to do all of your manipulations in a single cell!

Note that this may not be as well-behaved as I would like it to be. I'm still testing configuration&mdash;if you have
any issues be sure to [file them]((https://github.com/ResidentMario/missingno/issues)).
-->

## Contributing

For thoughts on features or bug reports see the [bug tracker](https://github.com/ResidentMario/missingno/issues). If 
you're interested in contributing to this library, see details on doing so in the `CONTRIBUTING.md` file in this 
repository.

I'm keen in hearing feedback&mdash;reach out to me at `aleksey@residentmar.io` if you have it.