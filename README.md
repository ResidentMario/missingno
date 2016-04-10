# missingno

Messy datasets? Missing values? `missingno` provides a small toolset of flexible and easy-to-use missing data
visualizations and utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of
your dataset. It's built using `matplotlib`, so it's fast, and takes any `pandas.DataFrame` input that you throw at
it, so it's flexible. Just `pip install missingno` to get started.

# Quickstart

All examples use the [NYPD Motor Vehicle Collisions Dataset](https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95).

First up, the `missingno` nullity matrix is a data-dense display which lets you quickly visually pick out patterns in
 data completion:

    >>> import missingno as msno
    >>> msno.matrix(data.sample(250))

![alt text][two_hundred_fifty]

At a glance, date, time, the distribution of injuries, and the contribution factor of the first vehicle appear to be
completely populated, while geographic information seems mostly complete, but spottier.

[two_hundred_fifty]: http://i.imgur.com/DdepYwr.png

The missingno correlation heatmap lets you measure how strongly the presence of one variable positively or negatively
affect the presence of another:

    >>> msno.heatmap(data)

![alt text][heatmap]

[heatmap]: http://i.imgur.com/VOS6dqf.png

Hmm. It seems that reports which are filed with an `OFF STREET NAME` variable are less likely to have complete
geographic data.

Finally the dendrogram view allows you to more fully correlate variable completion, revealing trends deeper than the
pairwise ones visible in the correlation heatmap:

    >>> msno.dendrogram(dat)

![alt text][dendrogram]

[dendrogram]: http://i.imgur.com/6ZBC4af.png

`missingno` also provides utility functions for filtering records in your dataset based on completion. These are
useful in particular for filtering through and drilling down into particularly large datasets whose data missingness
issues might otherwise be very hard to visualize or understand.

    >>> filtered_data = msno.nullity_filter(data, filter='bottom', n=5, p=0.9) # or filter='top'
    >>> filtered_sorted_data = msno.nullity_sort(data, sort='descending') # or sort='ascending'
    >>> msno.matrix(filtered_data.sample(250))

![alt text][matrix_sorted_filtered]

[matrix_sorted_filtered]: http://i.imgur.com/qL6zNQj.png

# Going further

<div style="background:#ddd; font-weight:bold; padding:25px;">

[For more on missingno functions check out my tutorial on working with missing data in Python!](http://nbviewer.jupyter.org/github/ResidentMario/python-missing-data/blob/master/missing-data.ipynb)

[For more on this module's ideation check out this post on my personal blog](http://www.residentmar.io/2016/03/28/missingno.html).
</div>



# Contributing

Bugs? Thoughts? Feature requests? [Throw them at the bug tracker and I'll take a look](https://github.com/ResidentMario/missingno/issues).
As always I'm very interested in hearing feedback: you can also reach out to me at `aleksey@residentmar.io`.
