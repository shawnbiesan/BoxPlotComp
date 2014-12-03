__author__ = 'sbiesan'

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import numpy as np
import Stemmer



class StemmedTfidfVectorizer(TfidfVectorizer):
    """
    dfgdg
    """

    def build_analyzer(self):
         stemmer = Stemmer.Stemmer('en')
         analyzer = super(TfidfVectorizer, self).build_analyzer()
         return lambda doc: stemmer.stemWords(analyzer(doc))

class CustomTransformer(object):
    def __init__(self, cols):
        self.cols = cols
        self.model = dict()

    def transform(self, X):
        #print "entered transform"
        X[self.cols] = X[self.cols].fillna('')
        arrays = tuple(self.model[col].transform(X[col]) for col in self.cols)
        result = sp.sparse.hstack(arrays).tocsr()
        #print result.shape
        #print "finished transform"
        return result


    def fit(self, X, y=None):
        #print "entered fit"
        X[self.cols] = X[self.cols].fillna('')
        for col in self.cols:
            #self.model[col] = StemmedTfidfVectorizer(stop_words='english', ngram_range=(1,4))
            self.model[col] = TfidfVectorizer(stop_words='english',max_features=200)
            self.model[col].fit(X[col])
        #print "finished fit"
        return self


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
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

    def transform(self, data_dict):
        #print "entered itemselector"
        result = data_dict[self.key]
        #print result.shape
        #print "finished itemselector"
        return result.reshape(result.shape[0], 1)
