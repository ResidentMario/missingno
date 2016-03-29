import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt


def missingno(df, figsize=(20, 10), width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
              fontsize=16, labels=True):
    # The height and width parameters are the number of rows and the number of columns in the dataframe, respectively.
    height = df.shape[0]
    width = df.shape[1]
    # Z is the color-mask array.
    Z = df.isnull().values

    # G is a NxNx3 matrix
    G = np.zeros((height, width, 3))

    # Apply the Z color-mask to set the RGB of each pixel.
    G[Z > 0.5] = [1, 1, 1]
    G[Z < 0.5] = color

    # Set up the matplotlib grid layout.
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
    gs.update(wspace=0.08)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    # Create the nullity plot.
    ax0.imshow(G, interpolation='none')

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(b=False)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    # Remove the black border.
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)

    # Set up and rotate the column ticks.
    if labels:
        ha = 'left'
        ax0.set_xticks(list(range(0, width)))
        ax0.set_xticklabels(list(df.columns), rotation=45, ha=ha, fontsize=fontsize)
    else:
        ax0.set_xticks([])

    # Set up the two top-bottom row ticks.
    ax0.set_yticks([0, df.shape[0] - 1])
    ax0.set_yticklabels([1, df.shape[0]], fontsize=20)
    # Create the inter-column vertical grid.
    in_between_point = [x + 0.5 for x in range(0, width - 1)]
    for in_between_point in in_between_point:
        ax0.axvline(in_between_point, linestyle='-', color='white')

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

    plt.show()
