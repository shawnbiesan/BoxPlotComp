"""
Main Entry point to run experiments
"""

__author__ = 'sbiesan'

import pandas as pd
from sklearn import cross_validation
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import hamming_loss, log_loss, roc_auc_score
from sklearn.decomposition import truncated_svd
from column_info import outputs, text_features, num_features
from sklearn.grid_search import GridSearchCV
from multi_log_loss import multi_multi_log_loss, BOX_PLOTS_COLUMN_INDICES
import datetime

from CustomClasses import CustomTransformer, ItemSelector, CustomPipeline

import numpy as np

sample = pd.read_csv('SubmissionFormat.csv')


def pick_best_features(train, test):
    """
    Grid search to find best features. TODO refactor
    :param train: train data
    :param test: test data
    :return:
    """
    for output in outputs:
        pipe_clf = CustomPipeline.get_transforms()

        clf = SVC(probability=True, kernel='linear')

        tuned_parameters = [{'C': [1, 10, 100]}]
        score = 'log_loss'
        tran_x = pipe_clf.fit_transform(train)
        grid = GridSearchCV(clf, tuned_parameters, cv=5, scoring=score)
        grid.fit(tran_x, test[output])
        print "best param: "
        print grid.best_estimator_

def top_words(clf):
    """
    placeholder to be used to print top words, not sure how to use yet since there are multiple text
    columns
    :param clf:
    :return:
    """
    for column_ in clf.steps[0][1].model.keys():
        feature_names = clf.steps[0][1].model[column_].get_feature_names() ## select k best
    top10 = np.argsort(clf.steps[1][1].scores_)
    for entry in top10:
        print feature_names[entry]

def sample_data_random(df, percent):
    """
    Sampling that forces all labels to be available for cross validation
    This makes it so log_loss doesn't error
    :param df: pandas data frame
    :param percent: percent to random sample
    :return:
    """
    indices = np.random.choice(df.index, int(df.shape[0] * percent), replace=False)
    return df.loc[indices]


def sample_data(df, percent):
    """
    Sampling that forces all labels to be available for cross validation
    This makes it so log_loss doesn't error when the sample size is small
    :param df: pandas data frame
    :param percent: percent to random sample before manual pull
    :return:
    """
    indices = np.random.choice(df.index, int(df.shape[0] * percent), replace=False)
    #indices = np.random.choice(df.index, 500, replace=False)
    for output in outputs:
        current_labels = pd.unique(df[output])
        for label in current_labels:
            x = df[output]
            x1 = x[x == label]
            selected = x1[~x1.index.isin(indices)].index[0:50]
            indices = np.concatenate((indices, selected))
    return df.loc[indices]

def tester(text_, column_):
    """
    Deprecated method to try to see if text matches a column
    :param text_: text to check
    :param column_: column iterable to check
    :return:
    """
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

def validate_model(df):
    """
    Validates model with averaged log loss from sklearn
    :param df: pandas data frame
    :return:
    """
    train = sample_data_random(df, 0.15)
    labels = train[outputs]

    #Simple K-Fold cross validation. 5 folds.


    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = defaultdict(list)
    sum_mean = 0
    sum_std = 0
    #train = add_features(train)

    for output in outputs:
        cv = cross_validation.StratifiedKFold(labels[output])

        for traincv, testcv in cv:
            pipe_clf = CustomPipeline.get_pipeline()

            train_sample = train.reset_index().loc[traincv]
            train_test = labels.reset_index()[output].loc[traincv]

            test_sample = train.reset_index().loc[testcv]
            test_test = labels.reset_index()[output].loc[testcv]

            print outputs
            print output
            print "------------------------"

            pipe_clf.fit(train_sample, train_test)
            #top_words(pipe_clf)
            preds = pipe_clf.predict_proba(test_sample)
            results[output].append(log_loss(test_test, preds))


    for result in results:
        sum_mean += np.array(results[result]).mean()
        sum_std += np.array(results[result]).std()
        print "Results %s: " % (result,) + str( np.array(results[result]).mean()) + " " + str(np.array(results[result]).std())
    print "Final: %s %s" %(1.0 * sum_mean / len(results), 1.0 * sum_std / len(results))
    f = open("validate_" + str(datetime.datetime.now()), 'wt')
    f.write("Final: %s %s" %(1.0 * sum_mean / len(results), 1.0 * sum_std / len(results)))


def validate_model_real(df):
    """
    Cross Validation with the provided metric from the website of the competition
    :param df: pandas data frame
    :return:
    """
    train = sample_data_random(df, 0.10)
    labels = train[outputs]
    #Simple K-Fold cross validation. 3 folds.


    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    #train = add_features(train)

    cv = cross_validation.StratifiedKFold(labels[outputs[0]])

    for traincv, testcv in cv:
        combined_pred = []
        combined_real = []
        for output in outputs:
            pipe_clf = CustomPipeline.get_pipeline()

            train_sample = train.reset_index().loc[traincv]
            train_test = labels.reset_index()[output].loc[traincv]

            test_sample = train.reset_index().loc[testcv]
            test_test = labels.reset_index()[output].loc[testcv]

            print outputs
            print output
            print "------------------------"

            pipe_clf.fit(train_sample, train_test)
            combined_pred.append(pipe_clf.predict_proba(test_sample))
            combined_real.append(pd.get_dummies(test_test).values)
        np_pred = np.concatenate(combined_pred, axis=1)
        np_real = np.concatenate(combined_real, axis=1)
        print np_pred.shape
        print np_real.shape

        results.append(multi_multi_log_loss(np_pred, np_real, BOX_PLOTS_COLUMN_INDICES))
        print results


    print np.array(results).mean()
    print np.array(results).std()

def test_model(df):
    """
    Currently just trains on a randomly selected 25% sample of the data
    and tests it on another 25% (may intersect)
    Should rethink this, and have a static hold out?
    :param df:
    :return:
    """
    results = defaultdict(int)

    train = sample_data_random(df, 0.50)
    labels_train = train[outputs]

    test = sample_data_random(df, 0.25)
    labels_test = test[outputs]

    models = dict()

    for output in outputs:
        pipe_clf = CustomPipeline.get_pipeline()
        print "Fitting %s" % (output,)
        pipe_clf.fit(train, labels_train[output])
        preds = pipe_clf.predict_proba(test)
        results[output] = log_loss(labels_test[output], preds)
        models[output] = pipe_clf

    sum_result = 0
    for result in results:
        print "Results %s: " % (result,) + str(results[result])
        sum_result += results[result]
    print 1.0 * sum_result / len(results)
    f = open("test_" + str(datetime.datetime.now()), 'wt')
    f.write(str(1.0 * sum_result / len(results)))
    return models


train = pd.read_csv('TrainingData.csv')
train[num_features] = train[num_features].fillna(0.0)

test = pd.read_csv('TestData.csv')
test[num_features] = test[num_features].fillna(0.0)


#validate_model_real(train, train[outputs])
validate_model(train)
models = test_model(train)
#pick_best_features(train, train[outputs])

for output in outputs:
    print "predicting %s" % (output,)
    result = models[output].predict_proba(test)
    classes_ = models[output].steps[2][1].classes_
    sample[[output + '__' + entry for entry in classes_]] = result

sample.to_csv("resultmixed.csv", index=False)