"""Functions that process data for modelling."""

from sklearn.base import BaseEstimator, TransformerMixin


def encode_categorical(data, features, target):
    """
    Encode categorical features based on effect on target.

    Map each level of the categorical variables to an integer, based on
    how the levels affect the percentage of positive examples in target.
    Encoded features are added to data

    Parameters
    ----------
    data: pandas.DataFrame
        Data frame containing the data.
    features: List[str]
        Names of the columns of data containing the categorical features
        to encode.
    target: str
        Name of the target column.

    Returns
    -------
    pandas.DataFrame
        data with encoded features added

    """
    out_df = data.copy()
    for feature in features:
        encoding = (
            out_df.groupby(feature)
                  .agg({target: 'mean'})
                  .sort_values(target)
                  .reset_index()
                  .reset_index()
                  .set_index(feature)
        )
        level_map = encoding['index'].to_dict()
        out_df[f'{feature}_encoded'] = out_df[feature].map(level_map).astype(float)
    return out_df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Wrap encode_categorical into a scikit-learn estimator.

    This enables the categorical encoding to be used in a Pipeline object.

    """

    def __init__(self, features_to_encode, target, features_to_return):
        self.features_to_encode = features_to_encode
        self.target = target
        self.features_to_return = features_to_return

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        X_cat = X.copy()
        X_cat = encode_categorical(X_cat, self.features_to_encode, self.target)
        return X_cat[self.features_to_return]
