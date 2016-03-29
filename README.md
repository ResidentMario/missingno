# missingno

Messy datasets? Missing values? `missingno` provides a flexible and easy-to-use missing data matrix (nullity matrix?)
visualization that allows you to get a quick visual summary of the completeness (or lack thereof) of your dataset.
It's built using `matplotlib`, so it's fast, and takes any `DataFrame` input that you throw at it, so it's flexible.
Just `pip install missingno` to get started.

Here is a 100-record sample from the [NYPD Motor Vehicle Collisions Dataset](https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95):

    >>> from missingno import missingno
    >>> missingno(df.sample(100))

![alt text][one_hundred]

At a glance, date, time, the distribution of injuries, and the contribution factor of the first vehicle appear to be
completely populated, while geographic information seems mostly complete, but spottier. The completion sparkgraph at
right demonstrates a strong clustering about 20 filled values.

Here's what happens when we throw 1000 records at it:

    >>> missingno(df.sample(1000))

![alt text][one_thousand]

[one_hundred]: http://i.imgur.com/g8Rserl.png
[one_thousand]: http://i.imgur.com/y2RfKnS.png

`missingno` provides the following optional arguments (defaults indicated), all of which are passed to `matplotlib`
under the hood:

* `figsize=(20, 10)` --- Adjusts the aspect ratio and size of the graph.
* `width_ratios=(15, 1)` --- Adjusts the relative sizes of the main plot and the sparkgraph.
* `color=(0.25, 0.25, 0.25)` --- Adjusts the color of the filled matrix entries and of the sparkline. Note that
`matplotlib` (atypically) represents RGB values in terms of a fraction out of one! So e.g. `0 = 0` and `1 = 255`; to
input your typical RGB value (`122` for instance) pass `122/255` instead.
* `fontsize=16` --- Adjusts the font-sizes used for display. Essential for datasets with lots of columns or for small
displays.
* `labels=True` --- Set this to `False` to turn off the y-axis labels. If you have a huge number of columns this is
probably necessary.

[For more on this module's ideation check out this post on my personal blog](http://www.residentmar.io/2016/03/28/missingno.html).

[If you like this project be sure to also check out the pandas-profiling module](https://github.com/JosPolfliet/pandas-profiling).

Bugs? Thoughts? Feature requests? [Throw them at the bug tracker and I'll take a look](https://github.com/ResidentMario/missingno/issues).
As always I'm very interested in hearing feedback: you can also reach out to me at `aleksey@residentmar.io`.
