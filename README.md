# missingno

Messy datasets? Missing values? `missingno` provides a small toolset of flexible and easy-to-use missing data
visualizations and utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of
your dataset. It's built using `matplotlib`, so it's fast, and takes any `pandas.DataFrame` input that you throw at
it, so it's flexible. Just `pip install missingno` to get started.

## Quickstart

All examples use the [NYPD Motor Vehicle Collisions Dataset](https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95).

I take **nullity** to mean whether a particular variable is filled in or not.

### Matrix

The `msno.matrix` nullity matrix is a data-dense display which lets you quickly visually pick out patterns in
 data completion.

    >>> import missingno as msno
    >>> %matplotlib inline
    >>> msno.matrix(data.sample(250))

![alt text][two_hundred_fifty]

[two_hundred_fifty]: http://i.imgur.com/DdepYwr.png

At a glance, date, time, the distribution of injuries, and the contribution factor of the first vehicle appear to be
completely populated, while geographic information seems mostly complete, but spottier.

The sparkline at right summarizes the general shape of the data completeness and points out the maximum and minimum
rows.

### Heatmap

The missingno correlation heatmap lets you measure how strongly the presence of one variable positively or negatively
affect the presence of another:

    >>> msno.heatmap(data)

![alt text][heatmap]

[heatmap]: http://i.imgur.com/ESsZRlY.png

Hmm. It seems that reports which are filed with an `OFF STREET NAME` variable are less likely to have complete
geographic data.

Nullity correlation ranges from `-1` (if one variable appears the other definitely does not) to `0` (variables appearing
or not appearing have no effect on one another) to `1` (if one variable appears the other definitely also does).
Entries marked `<1` or `>-1` are have a correlation that is close to being exactingly negative or positive, but is
still not quite perfectly so. This points to a small number of records in the dataset which are erroneous. For
example, in this dataset the correlation between `VEHICLE CODE TYPE 3` and `CONTRIBUTING FACTOR VEHICLE 3` is `<1`,
indicating that, contrary to our expectation, there are a few records which have one or the other, but not both.
These cases will require special attention.

Note that variables with a variance of zero (that is, variables which are always full or always empty) have no
meaningful correlation and so are silently removed from the visualization&mdash;in this case for instance the
datetime and injury number columns, which are completely filled, are not included.

The heatmap works great for picking out data completeness relationships between variable pairs, but its visual power
is limited when it comes to larger relationships.


### Dendrogram

The dendrogram allows you to more fully correlate variable completion, revealing trends deeper than the pairwise
ones visible in the correlation heatmap:

    >>> msno.dendrogram(data)

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

    >>> filtered_data = msno.nullity_filter(dat, filter='bottom', n=15, p=0.999) # or filter='top'
    >>> msno.matrix(filtered_data.sample(250))

![alt text][matrix_filtered]

[matrix_filtered]: http://i.imgur.com/UF6hmL8.png

`nullity_sort()` simply reshuffles your rows by completeness, in either `ascending` or `descending` order. Since it
doesn't affect the underlying data it's mainly useful for `matrix` visualization:


    >>> sorted_data = msno.nullity_sort(data, sort='descending') # or sort='ascending'
    >>> msno.matrix(sorted_data.sample(250))

![alt text][matrix_sorted]

[matrix_sorted]: http://i.imgur.com/qL6zNQj.png

One final note. These methods also work inline within the visualization methods themselves. For instance, the
following is perfectly valid:

    >>> msno.matrix(data.sample(250), filter='top', n=5, p=0.9, sort='ascending')

## Visual configuration

### Lesser parameters

Each of the visualizations provides a further set of lesser configuration parameters for visually tweaking the display.

`matrix`, `heatmap`, and `dendrogram` all provide:

* `figsize`: The size of the figure to display. This is a `matplotlib` parameter which defaults to `(20, 12)`.
* `fontsize`: The figure's font size. A default size configured on the fly based on the `figsize` and the number of
columns.
* `labels`: Whether or not to display the column names. Defaults to `True`. Needs to be turned off for large datasets.

`matrix` also provides:
* `sparkline`: Set this to `False` to not draw the sparkline.
* `width_ratios`: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15,
    1)`. Does nothing if `sparkline=False`.
* `color`: The color of the filled columns. Defaults to `(0.25, 0.25, 0.25)`.

`heatmap` also provides:
* `cmap`: What `matplotlib` [colormap](http://matplotlib.org/users/colormaps.html) to use. Defaults to `RdBu`.


`dendrogram` also provides:
* `orientation` The [orientation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram)
of the dendrogram. Defaults to `top` if `<=50` columns and
`left` if there are more.
* `method`: The [linkage method](http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage) `scipy.hierarchy` uses for clustering.
`average` is the default argument.


### Advanced configuration
If you are not satisfied with these admittedly basic configuration parameters, the display can be further manipulated
in any way you like using `matplotlib` post-facto.

Every `missingno` visualization method returns its underlying `matplotlib.figure.Figure` object. Anyone with
sufficient knowledge of `matplotlib` operations and [the missingno source code](https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py)
can then tweak the display to their liking.

Note that this may not be as well-behaved as I would like it to be. I'm still testing configuration&mdash;if you have
any issues be sure to [file them]((https://github.com/ResidentMario/missingno/issues)).

## Further reading

If you're interested in learning more about working with missing data in Python check out [my tutorial on the
subject](http://nbviewer.jupyter.org/github/ResidentMario/python-missing-data/blob/master/missing-data.ipynb).

For more on this module's ideation check out [this post on my personal blog](http://www.residentmar.io/2016/03/28/missingno.html).


## Contributing

Bugs? Thoughts? Feature requests? [Throw them at the bug tracker and I'll take a look](https://github.com/ResidentMario/missingno/issues).

As always I'm very interested in hearing feedback: you can also reach out to me at `aleksey@residentmar.io`.