__author__ = 'sbiesan'

import pandas as pd
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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


import numpy as np

sample = pd.read_csv('SubmissionFormat.csv')
stemmer = Stemmer.Stemmer('en')


def sample_data(df):
    return df.loc[np.random.choice(df.index, df.shape[0] / 3, replace=False)]

def string_concat_columns(df):
    df['mew'] =      str(df['Facility_or_Department'] + ' ' +
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
    return df

class StemmedTfidfVectorizer(TfidfVectorizer):
    """
    dfgdg
    """

    def build_analyzer(self):
         analyzer = super(TfidfVectorizer, self).build_analyzer()
         return lambda doc: stemmer.stemWords(analyzer(doc))

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
        clf = SGDClassifier(alpha=.01, loss='log')
        for traincv, testcv in cv:

            transformed_x = tfidf.fit_transform(train.values[traincv])
            transformed_y = tfidf.fit_transform(train.values[testcv])
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

train[text_features] = train[text_features].fillna('NA')
train = string_concat_columns(train)
train['LenAll'] = len(train['mew'])
train[num_features] = train[num_features].fillna(0)


test[text_features] = test[text_features].fillna('NA')
test = string_concat_columns(test)
test['LenAll'] = len(test['mew'])
test[num_features] = test[num_features].fillna(0)


tfidf = StemmedTfidfVectorizer(stop_words='english')

transformed_x = tfidf.fit_transform(train['mew'])
transformed_y = tfidf.transform(test['mew'])



validate_model(train['mew'], train[outputs])


#for output in outputs:
#    clf = SGDClassifier(alpha=.01, loss='log')
#    clf.fit(transformed_x, train[output])
#    result = clf.predict_proba(transformed_y)
#    sample[[output + '__' + entry for entry in clf.classes_]] = result

#sample.to_csv("resultmixed.csv", index=False)