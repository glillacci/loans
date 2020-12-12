"""Convenience functions for plotting."""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple


def percentage_stacked_bar_plot(groups: str, outcome: str, data: pd.DataFrame,
                                baseline: float = 0.7, bar_width: float = 0.85) -> None:
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


def compute_precision_recall_curve(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline,
                                   splitter: StratifiedShuffleSplit,
                                   thresholds: np.ndarray = np.arange(0, 1, 0.05)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average precision-recall curve over random train/test splits.

    Parameters
    ----------
    X: pandas.DataFrame
        Features to use for model training and testing.
    y: pandas.Series
        Labels to use for model training and testing.
    pipeline: sklearn.pipeline.Pipeline
        Scikit learn Pipeline containing the model to test.
    splitter: sklearn.model_selection.StratifiedShuffleSplit
        Iterable object that generates random train/test splits.
    thresholds: np.ndarray
        The threshold values to use for the calculation.

    Returns
    -------
    precisions: np.array
        Mean of precision scores by threshold over the splits.
    recalls: np.array
        Mean of recall scores by threshold over the splits.

    """
    precisions = np.zeros((splitter.n_splits, len(thresholds)))
    recalls = np.zeros((splitter.n_splits, len(thresholds)))

    for (i, (train_IDX, test_IDX)) in enumerate(splitter.split(X, y)):
        X_train = X.loc[train_IDX]
        y_train = y.loc[train_IDX]

        pipeline.fit(X_train, y_train)

        for j, threshold in enumerate(thresholds):
            X_test = X.loc[test_IDX]
            y_test = y.loc[test_IDX]
            predictions = (pipeline.predict_proba(X_test)[:, 1] > threshold).astype(int)
            precisions[i, j] = precision_score(y_test, predictions, zero_division=1)
            recalls[i, j] = recall_score(y_test, predictions)

    return np.mean(precisions, axis=0), np.mean(recalls, axis=0)
