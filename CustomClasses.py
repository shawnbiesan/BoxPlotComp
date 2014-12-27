"""
The intent of this file is to maintain any custom transforms/fits and the main custom pipeline
I use for my model
"""

__author__ = 'sbiesan'

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD
import scipy as sp
import numpy as np
import Stemmer
from column_info import outputs, text_features, num_features
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC


class StemmedTfidfVectorizer(TfidfVectorizer):
    """
    Custom Vectorizer for use with CustomTransformer. Additionally just stems words,
    saw no improvement so deprecated
    """

    def build_analyzer(self):
        """
        Overrides analyzer for TfidfVectorizer, currently only adds stem functionality
        :return: stemmed document
        """

        stemmer = Stemmer.Stemmer('en')
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: stemmer.stemWords(analyzer(doc))

class CustomTransformer(object):
    """
    Calls tfidf on each text column and then uses sparse hstack to combine them as feature matrix
    """

    def __init__(self, cols):
        self.cols = cols
        self.model = dict()

    def transform(self, X):
        """
        Transforms all 'columns' via tfidf
        :param X: pandas data frame
        :return: sparse matrix
        """
        X[self.cols] = X[self.cols].fillna('')
        arrays = tuple(self.model[col].transform(X[col]) for col in self.cols)
        result = sp.sparse.hstack(arrays).tocsr()
        return result


    def fit(self, X, y=None):
        """
        Fits separate Tfidf model to each respective column
        :param X: Pandas Data frame
        :param y: Not used
        :return: returns self
        """
        X[self.cols] = X[self.cols].fillna('')
        for col in self.cols:
            #self.model[col] = StemmedTfidfVectorizer(stop_words='english', ngram_range=(1,4))
            #self.model[col] = TfidfVectorizer(stop_words='english', ngram_range=(1,4))
            self.model[col] = TfidfVectorizer(ngram_range=(1,4))
            self.model[col].fit(X[col])
        #print "finished fit"
        return self


class TextFeatures(object):
    """
    Transformation class to be used for sklearn pipeline that gets normalized length of text columns
    as a feature
    """

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        X[self.cols] = X[self.cols].fillna('')
        output_columns = []
        for col in self.cols:
            col_name = 'Len_%s' % (col,)
            X[col_name] = X.apply(lambda row: len(row[col]), axis=1)
            output_columns.append(col_name)
            X[col_name] = (X[col_name] - X[col_name].mean()) / (X[col_name].max() - X[col_name].min())
        return X[output_columns]


    def fit(self, X, y=None):
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
        result = data_dict[self.key]
        result = (result - result.mean()) / (result.max() - result.min())
        return result.reshape(result.shape[0], 1)

class CustomPipeline(object):
    """
    Centralized place for changing pipeline info, sort of factory like.
    """
    @classmethod
    def get_pipeline(cls):
        pipe_clf = Pipeline([
        ('Features', FeatureUnion([
                    ('AddedFeatures', TextFeatures(text_features)),
                    ('text', CustomTransformer(text_features)),
                    ('fte', ItemSelector('FTE')),
                    ('total', ItemSelector('Total')),
                ])),
        ('svd', TruncatedSVD(n_components=400)),
        #('log', LogisticRegression(C=10)),
        ('sgd', SGDClassifier(alpha=.001, loss='log'))
        ])
        return pipe_clf

    @classmethod
    def get_transforms(cls):
        full = cls.get_pipeline()
        return Pipeline([step for i, step in enumerate(full.steps) if i != len(full.steps) - 1 ])