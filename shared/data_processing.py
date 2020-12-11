"""Functions to handle data."""


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
    for feature in features:
        encoding = (
            data.groupby(feature)
                .agg({target: 'mean'})
                .sort_values(target)
                .reset_index()
                .reset_index()
                .set_index(feature)
        )
        level_map = encoding['index'].to_dict()
        data[f'{feature}_encoded'] = data[feature].map(level_map)
    return data
