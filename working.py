__author__ = 'sbiesan'

import pandas as pd
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from collections import defaultdict
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.corpus import stopwords
import Stemmer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import hamming_loss, log_loss
from sklearn.decomposition import truncated_svd
import scipy as sp

import numpy as np

sample = pd.read_csv('SubmissionFormat.csv')
stemmer = Stemmer.Stemmer('en')


def sample_data(df):
    return df.loc[np.random.choice(df.index, df.shape[0] / 3, replace=False)]

def tester(text_, column_):
    num = 0
    for entry in column_:
        for word in entry.split():
            if word in text_:
                num+=1
    return num


def string_concat_columns(df):
    result = str(df['Facility_or_Department'] + ' ' +
                    df['Function_Description'] + ' ' +
                    df['Fund_Description'] + ' ' +
                    df['Job_Title_Description'] + ' ' +
                    df['Location_Description'] + ' ' +
                    df['Object_Description'] + ' ' +
                    df['Position_Extra'] + ' ' +
                    df['Program_Description'] + ' ' +
                    df['SubFund_Description'] + ' ' +
                    df['Sub_Object_Description'] + ' ' +
                    df['Text_1'] + ' ' +
                    df['Text_2'] + ' ' +
                    df['Text_3'] + ' ' +
                    df['Text_4'] + ' ' )
    return result

class StemmedTfidfVectorizer(TfidfVectorizer):
    """
    dfgdg
    """

    def build_analyzer(self):
         analyzer = super(TfidfVectorizer, self).build_analyzer()
         return lambda doc: stemmer.stemWords(analyzer(doc))

class BagMultiColumn(object):

    def __init__(self):
        pass

    def transform(self, df):
        tmp = self.tfidf.transform(df['mew'])
        return tmp

    def fit(self, df, y=None):
        print "called fit"

        self.tfidf = StemmedTfidfVectorizer(stop_words='english', ngram_range=(1,2))
        self.tfidf.fit(df['mew'])

        return self

def validate_model(train, labels):

    #Simple K-Fold cross validation. 5 folds.


    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = defaultdict(list)
    sum_mean = 0
    sum_std = 0

    for output in outputs:
        cv = cross_validation.StratifiedKFold(labels[output])
        tfidf = StemmedTfidfVectorizer(stop_words='english')
        svd = TruncatedSVD(n_components=100)
        clf = SGDClassifier(alpha=.01, loss='log', penalty='l2')
        for traincv, testcv in cv:

            transformed_x = tfidf.fit_transform(train['mew'].values[traincv])
            transformed_x = svd.fit_transform(transformed_x)
#            joined_x = np.hstack([transformed_x, train['LenAll'].values[traincv]])

            transformed_y = tfidf.fit_transform(train['mew'].values[testcv])
            transformed_y = svd.transform(transformed_y)
            #joined_y = np.hstack([transformed_y, train['LenAll'].values[testcv]])



            print outputs
            print output
            print "------------------------"

            clf.fit(transformed_x, labels[output].values[traincv])


            #print pd.unique(labels[output])
            #print pd.unique(clf.predict(train.values[testcv]))
            #print clf.predict(train.values[testcv])
            #results[output].append(hamming_loss(labels[output].values[testcv], clf.predict(transformed_y)))
            results[output].append(log_loss(labels[output].values[testcv], clf.predict_proba(transformed_y)))
            #

    for result in results:
        sum_mean += np.array(results[result]).mean()
        sum_std += np.array(results[result]).std()
        print "Results %s: " %(result,) + str( np.array(results[result]).mean()) + " " + str(np.array(results[result]).std())
    print "Final: %s %s" %(1.0 * sum_mean/ len(outputs), 1.0 * sum_std / len(outputs))

train = pd.read_csv('TrainingData.csv')
test = pd.read_csv('TestData.csv')
outputs = [
    'Function',
    'Object_Type',
    'Operating_Status',
    'Position_Type',
    'Pre_K',
    'Reporting',
    'Sharing',
    'Student_Type',
    'Use',
]

text_features = [
    'Facility_or_Department',
    'Function_Description',
    'Fund_Description',
    'Job_Title_Description',
    'Location_Description',
    'Object_Description',
    'Position_Extra',
    'Program_Description',
    'SubFund_Description',
    'Sub_Object_Description',
    'Text_1',
    'Text_2',
    'Text_3',
    'Text_4',
]

num_features = [
    'FTE',
    'Total',
    'LenAll',
]


train = sample_data(train)
columns_ = pd.unique(train.Function)

train[text_features] = train[text_features].fillna('')
train['mew'] = train.apply(lambda row: string_concat_columns(row), axis=1)
train['LenAll'] = len(train['mew'])
train['WeightedWordCounts'] = train.apply(lambda row: tester(row['mew'], columns_), axis=1)
#train[num_features] = train[num_features].fillna(0)


test[text_features] = test[text_features].fillna('')
test['mew'] = test.apply(lambda row: string_concat_columns(row), axis=1)
test['LenAll'] = len(test['mew'])
test['WeightedWordCounts'] = test.apply(lambda row: tester(row['mew'], columns_), axis=1)
#test[num_features] = test[num_features].fillna(0)


#validate_model(train, train[outputs])


tfidf = StemmedTfidfVectorizer(stop_words='english')

transformed_x = tfidf.fit_transform(train['mew'])
m = sp.sparse.csr_matrix(train.WeightedWordCounts.values.T)
transformed_x = sp.sparse.hstack((transformed_x, m.T))
transformed_y = tfidf.transform(test['mew'])
n = sp.sparse.csr_matrix(test.WeightedWordCounts.values.T)
transformed_y = sp.sparse.hstack((transformed_y, n.T))

for output in outputs:
#    #clf = SGDClassifier(alpha=.01, loss='log')
    clf = SGDClassifier(alpha=.01, loss='log', penalty='l2')
    clf.fit(transformed_x, train[output])
    result = clf.predict_proba(transformed_y)
    sample[[output + '__' + entry for entry in clf.classes_]] = result

sample.to_csv("resultmixed.csv", index=False)