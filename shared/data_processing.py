"""Classes that process the data using Scikit-learn Pipeline objects."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding of categorical variables.

    Map each level of the categorical variables to an integer. The integer values
    are increasing with the percentage of positive examples found in the level.

    Parameters
    ----------
    features_to_encode: List[str]
        Names of the categorical features to be encoded this way.

    """

    def __init__(self, features_to_encode: List[str], mode: str = 'fraction') -> None:
        self.features_to_encode = features_to_encode
        self.mode = mode
        self.mappings = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CategoricalEncoder':
        """
        Learn mapping from categorical levels to integers.

        """
        for feature in self.features_to_encode:
            feature_with_target = pd.concat([X[feature], y], axis=1)
            encoding = (
                feature_with_target.groupby(feature)
                                   .agg({y.name: 'mean'})
                                   .sort_values(y.name, ascending=False)
                                   .reset_index()
                                   .reset_index()
                                   .set_index(feature)
            )
            if self.mode == 'fraction':
                level_map = encoding[y.name].to_dict()
            else:
                level_map = encoding['index'].to_dict()
            self.mappings[feature] = level_map
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Apply the mapping to the categorical features.

        """
        if self.mappings is None:
            raise ValueError('Categorical mapping has not been learnt.')

        X_out = X.copy()

        for feature in self.features_to_encode:
            X_out[feature] = X_out[feature].map(self.mappings[feature])

        return X_out


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select features from a Pandas data frame."""

    def __init__(self, features_to_select: List[str]) -> None:
        self.features_to_select = features_to_select

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureSelector':
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return X[self.features_to_select]
