from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import six
from operator import itemgetter

class HashtagCountVectorizer(CountVectorizer):
    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        self._check_vocabulary()

        return ["#" + t for t, i in sorted(six.iteritems(self.vocabulary_),
                                     key=itemgetter(1))]

from sklearn.base import TransformerMixin

class ItemSelector(TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self
    
    def get_feature_names(self):
        return self.key

    def transform(self, data_dict):
        return data_dict[self.key].values.astype('str')