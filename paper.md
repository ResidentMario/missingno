---
title: 'Missingno: a missing data visualization suite'
tags:
  - missing data
  - data visualization
authors:
 - name: Aleksey Bilogur
   orcid: 0000-0002-0066-5825
   affiliation: 1
affiliations:
 - name: Independent
   index: 1
date: 28 January 2018
bibliography: paper.bib
---

# Summary

Algorithmic models and outputs are only as good as the data they are computed on. As the popular saying goes: garbage 
in, garbage out. In tabular datasets, it is usually relatively easily to, at a glance, understand patterns of 
missing data (or nullity) of individual rows, columns, and entries. However, it is far harder to see patterns in the
missingness of data that extend between them. Understanding such patterns in data is benefitial, if not outright 
critical, to most applications.

missingno is a Python package for visualizing missing data. It works by converting tabular data matrices into boolean 
masks based on whether individual entries contain data (which evaluates to true) or left empty (which evaluates to 
false). This "nullity matrix" is then exposed to user assessment through a variety of special-purpose data 
visualizations.

The simplest tools, the bar chart and matrix display, are literal translations of a data table's 
nullity matrix, and are effective for snapshotting general patterns.

![](http://i.imgur.com/DdepYwr.png)
![](http://i.imgur.com/lOTN3tm.png)

A heatmap provides a methodology for examining relationships within pairs of variables.

![](http://i.imgur.com/ESsZRlY.png)

Higher-cardinality data nullity correlations can be understood using a hierarchically clustered dendrogram.

![](http://i.imgur.com/6ZBC4af.png)

Finally, geospatial data dependencies are viewable using an approach based on the quadtree or convex hull algorithm.

![large-geoplot](http://i.imgur.com/4dtGhig.png)

The visualizations are consciously designed to be as effective as possible
at uncovering missing data patterns both between and within columns of data, and hence, to help its users build more 
effective data models and pipelines. At the same time the package is designed to be easy to use. The underlying 
packages involved (numpy, pandas, scipy, matplotlib, and seaborn) are familiar parts of the core scientific Python 
ecosystem, and hence very learnable and extensible. missingno works "out of the box" with a variety of data types and 
formats, and provides an extremely compact API.

# References