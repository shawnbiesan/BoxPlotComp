__author__ = 'sbiesan'

import pandas as pd
from sklearn import cross_validation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.corpus import stopwords
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import hamming_loss, log_loss
from sklearn.decomposition import truncated_svd
import scipy as sp
from column_info import outputs, text_features, num_features

from CustomClasses import CustomTransformer, ItemSelector

import numpy as np

sample = pd.read_csv('SubmissionFormat.csv')


def add_features(df):
    df[text_features] = df[text_features].fillna('')
    df['mew'] = df.apply(lambda row: string_concat_columns(row), axis=1)
    df['LenAll'] = len(df['mew'])

    for entry in outputs:
        columns_ = pd.unique(train[entry])
        columns_ = [column_.lower() for column_ in columns_]
        df['Weighted%s' % (entry,)] = df.apply(lambda row: tester(row['mew'], columns_), axis=1)
        num_features.append('Weighted%s' %(entry,))
    df[num_features] = df[num_features].fillna(0)
    return df



def sample_data(df):
    """
    Sampling that forces all labels to be available for cross validation
    This makes it so log_loss doesn't error
    :param df:
    :return:
    """
    indices = np.random.choice(df.index, df.shape[0] / 10, replace=False)
    for output in outputs:
        current_labels = pd.unique(df[output])
        for label in current_labels:
            x = df[output]
            x1 = x[x == label]
            selected = x1[~x1.index.isin(indices)].index[0:10]
            indices = np.concatenate((indices, selected))


    return df.loc[indices]

def tester(text_, column_):
    num = 0
    for entry in column_:
        for word in entry.split():
            if word.lower() in text_.lower():
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
    ' '.join(result.split())
    return result

def validate_model(train, labels):

    #Simple K-Fold cross validation. 5 folds.


    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = defaultdict(list)
    sum_mean = 0
    sum_std = 0
    train = add_features(train)

    for output in outputs:
        cv = cross_validation.StratifiedKFold(labels[output])
       # tfidf = CustomTransformer(text_features)
        #clf = SGDClassifier(alpha=.1, loss='log', penalty='l1')
        #clf = LogisticRegression(C=1000)
       # clf = MultinomialNB(alpha=.75)
        #clf = RandomForestClassifier(n_jobs=-1)
       # svd = TruncatedSVD(n_components=100)

        for traincv, testcv in cv:
            pipe_clf = Pipeline([
                    ('Features', FeatureUnion([
                                ('text', CustomTransformer(text_features)),
                                #('fte', ItemSelector('FTE')),
                                #('total', ItemSelector('Total')),
                            ])),
                    ('svd', TruncatedSVD(n_components=100)),
                    #('svd', PCA(n_components=100))
                    ('log', LogisticRegression(C=100)),
                    ])
            train_sample = train.reset_index().loc[traincv]
            train_test = labels.reset_index()[output].loc[traincv]

            test_sample = train.reset_index().loc[testcv]
            test_test = labels.reset_index()[output].loc[testcv]

        #    tfidf.fit(train_sample)

         #   transformed_x = tfidf.transform(train_sample)
          #  svd.fit(transformed_x)
          #  svd_x = svd.transform(transformed_x)

         #   transformed_y = tfidf.transform(test_sample)
         #   svd_y = svd.transform(transformed_y)



            print outputs
            print output
            print "------------------------"

            #clf.fit(transformed_x, labels[output].values[traincv])
            pipe_clf.fit(train_sample, train_test)


#            results[output].append(log_loss(labels[output].values[testcv], clf.predict_proba(transformed_y)))
            #print pd.unique(test_test)
            #print pd.unique(pipe_clf.predict(test_sample))
            results[output].append(log_loss(test_test, pipe_clf.predict_proba(test_sample)))


    for result in results:
        sum_mean += np.array(results[result]).mean()
        sum_std += np.array(results[result]).std()
        print "Results %s: " % (result,) + str( np.array(results[result]).mean()) + " " + str(np.array(results[result]).std())
    print "Final: %s %s" %(1.0 * sum_mean / len(outputs), 1.0 * sum_std / len(outputs))

train = pd.read_csv('TrainingData.csv')
test = pd.read_csv('TestData.csv')

train = sample_data(train)


#tfidf = CustomTransformer(text_features)

#tfidf.fit(train)
#transformed_x = tfidf.transform(train)

#transformed_y = tfidf.transform(test)
#train = tfidf.transform(train)

#test = tfidf.transform(test)

#train = add_features(train)
#test = add_features(test)

validate_model(train, train[outputs])

fdgsdg

for output in outputs:
    pipe_clf = Pipeline([
        ('Features', FeatureUnion([
                    ('text', CustomTransformer(text_features)),
                    #('fte', ItemSelector('FTE')),
                    #('total', ItemSelector('Total')),
                ])),
        ('svd', TruncatedSVD(n_components=100)),
        #('svd', PCA(n_components=100))
        #('log', LogisticRegression()),
        ('tree', RandomForestClassifier(n_jobs=-1))
        ])
    #clf = SGDClassifier(alpha=.01, loss='log')
    pipe_clf.fit(train, train[output])
 #   pipe_clf.fit(train, train[output])
    print "fit %s" %(output,)
    result = pipe_clf.predict_proba(test)
    classes_ = pipe_clf.steps[2][1].classes_
    sample[[output + '__' + entry for entry in classes_]] = result

sample.to_csv("resultmixed.csv", index=False)