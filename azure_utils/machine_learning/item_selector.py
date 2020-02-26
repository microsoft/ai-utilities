"""
ai-utilities - machine_learning/item_selector.py

From: http://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html

Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at provided
    key(s).

    The data are expected to be stored in a 2D data structure, where
    the first index is over features and the second is over samples,
    i.e.

    >> len(data[keys]) == n_samples

    Please note that this is the opposite convention to scikit-learn
    feature matrices (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[keys]).  Examples include: a dict of lists, 2D numpy array,
    Pandas DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample
    (e.g. a list of dicts).  If your data are structured this way,
    consider a transformer along the lines of
    `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    keys : hashable or list of hashable, required
        The key(s) corresponding to the desired value(s) in a mappable.

    """

    def __init__(self, keys):
        if isinstance(keys, list):
            if any([getattr(key, '__hash__', None) is None for key in keys]):
                raise TypeError('Not all keys are hashable')
        elif getattr(keys, '__hash__', None) is None:
            raise TypeError('keys is not hashable')
        self.keys = keys

    def fit(self, input_x, *args, **kwargs):
        """

        :param input_x: Set of items to fit with keys
        :return: self
        """
        if isinstance(self.keys, list):
            if not all([key in input_x for key in self.keys]):
                raise KeyError('Not all keys in data')
        elif self.keys not in input_x:
            raise KeyError('key not in data')
        return self

    def transform(self, data_dict, *args, **kwargs):
        """
        Transform data based on keys

        :param data_dict: Data to Transform
        :return: Transformed data
        """
        return data_dict[self.keys]

    def get_feature_names(self):
        """
        Get Feature Names

        :return: get keys
        """
        return self.keys
