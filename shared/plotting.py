"""Convenience functions for plotting."""

import numpy as np
from matplotlib import pyplot as plt


def percentage_stacked_bar_plot(groups, outcome, data, baseline=0.7, bar_width=0.85):
    """
    Generate a percentage stacked bar plot.

    Parameters
    ----------
    groups: str
        Name of the categorical variable in data to analyze.
    outcome: str
        Name of the label to cluster by, encoded as integers,
        1 for the positive class and 0 for the negative class.
    data: pandas.DataFrame
        Data frame containing the data.
    baseline: float
        Fraction of negative samples in the whole data set, to use as
        a reference for comparison.
    bar_width: float
        Relative width of the bars in the bar plot.

    """
    # Group counts by outcome
    green = data.loc[data[outcome] == 0, groups].value_counts().sort_index().values
    orange = data.loc[data[outcome] == 1, groups].value_counts().sort_index().values
    totals = data[groups].value_counts().sort_index()

    # From counts to percentage
    green_bars = green / totals.values
    orange_bars = orange / totals.values

    # Map bars to locations in increasing order of accept probability
    r = np.argsort(green_bars)
    loc = list(range(len(totals)))

    # Plot
    fig = plt.figure(figsize=(10, 5))
    plt.bar(loc, green_bars[r], color='#b5ffb9', edgecolor='white', width=bar_width)
    plt.bar(loc, orange_bars[r], bottom=green_bars[r], color='#f9bc86', edgecolor='white', width=bar_width)
    plt.hlines(baseline, min(r) - bar_width / 2, max(r) + bar_width / 2, linestyles='dashed')
    plt.title(groups)
    # Custom x axis
    names = totals.index
    plt.xticks(loc, names[r], rotation=30, ha='right')
