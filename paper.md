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
date: 13 Febuary 2018
bibliography: paper.bib
---

# Summary

Algorithmic models and outputs are only as good as the data they are computed on. As the popular saying goes: garbage 
in, garbage out. In tabular datasets, it is usually relatively easy to, at a glance, understand patterns of 
missing data (or nullity) of individual rows, columns, and entries. However, it is far harder to see patterns in the
missingness of data that extend between them. Understanding such patterns in data is beneficial, if not outright 
critical, to most applications.

missingno is a Python package for visualizing missing data. It works by converting tabular data matrices into boolean 
masks based on whether individual entries contain data (which evaluates to true) or left empty (which evaluates to 
false). This "nullity matrix" is then exposed to user assessment through a variety of special-purpose data 
visualizations.

The simplest tool, the bar chart, is a snapshot of column-level information:

![](https://i.imgur.com/2BxEfOr.png)

The matrix display provides a literal translations of a data table's 
nullity matrix. It is useful for snapshotting general patterns:

![](https://i.imgur.com/gWuXKEr.png)

A heatmap provides a methodology for examining relationships within pairs of variables.

![](https://i.imgur.com/JalSKyE.png)

Higher-cardinality data nullity correlations can be understood using a hierarchically clustered dendrogram:

![](https://i.imgur.com/oIiR4ct.png)

Finally, geospatial data dependencies are viewable using an approach based on the quadtree or convex hull algorithm:

![](https://i.imgur.com/0aaNa9Q.png)

The visualizations are consciously designed to be as effective as possible
at uncovering missing data patterns both between and within columns of data, and hence, to help its users build more 
effective data models and pipelines. At the same time the package is designed to be easy to use. The underlying 
packages involved (NumPy [@numpy], pandas [@pandas], SciPy [@scipy], matplotlib [@matplotlib], and seaborn [@seaborn]) are familiar parts of the core scientific Python 
ecosystem, and hence very learnable and extensible. missingno works "out of the box" with a variety of data types and 
formats, and provides an extremely compact API.

# References
