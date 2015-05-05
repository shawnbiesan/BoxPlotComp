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
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from multi_log_loss import multi_multi_log_loss, BOX_PLOTS_COLUMN_INDICES
import datetime
import time
from scipy.stats import uniform as sp_rand

from CustomClasses import CustomTransformer, ItemSelector, CustomPipeline

import numpy as np

sample = pd.read_csv('SubmissionFormat.csv')


def pick_best_features(df):
    """
    Grid search to find best features. TODO refactor
    :param train: train data
    :param test: test data
    :return:
    """

    #X = sample_data_random(df, .25)
    X = df[0:int(df.shape[0] * .25)]
    overfit_models = dict()
    for out in outputs:
        print out
        pipe_clf = CustomPipeline.get_transforms()

        clf = SGDClassifier(loss='log')

        tuned_parameters = {'alpha': sp_rand()}
        score = 'log_loss'
        tran_x = pipe_clf.fit_transform(X)
        grid = RandomizedSearchCV(clf, tuned_parameters, cv=5, scoring=score)
        grid.fit(tran_x, X[out])
        print grid.best_estimator_
        overfit_models[out] = grid.best_estimator_
    return overfit_models

def sample_data_random(df, percent):
    """
    Random sample data, without replacement
    :param df: pandas data frame
    :param percent: percent to random sample
    :return:
    """
    indices = np.random.choice(df.index, int(df.shape[0] * percent), replace=False)
    return df.loc[indices]


def sample_data_random_biased(df, percent):
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

def string_concat_columns(df):
    """
    Deprecated
    """
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
    #train = sample_data_random(df, 0.30)
    train = df[0:int(df.shape[0] * .30)] # Want determinism in model validation
    labels = train[outputs]



    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = defaultdict(list)
    sum_mean = 0
    sum_std = 0

    for output in outputs:
        cv = cross_validation.StratifiedKFold(labels[output])

        for traincv, testcv in cv:
            pipe_clf = CustomPipeline.get_transforms()

            train_sample = train.reset_index().loc[traincv]
            train_test = labels.reset_index()[output].loc[traincv]

            test_sample = train.reset_index().loc[testcv]
            test_test = labels.reset_index()[output].loc[testcv]

            print outputs
            print output
            print "------------------------"
            t0 = time.time()
            trans = pipe_clf.fit_transform(train_sample)
            trans_test = pipe_clf.transform(test_sample)

            #model1 = SGDClassifier(alpha=.0001, loss='log')
            model1 = LogisticRegression(C=10)
            model2 = RandomForestClassifier(n_jobs=-1, n_estimators=500)

            model1.fit(trans, train_test)
            model2.fit(trans, train_test)

            #top_words(pipe_clf)
            #preds = pipe_clf.predict_proba(test_sample)
           # preds = model1.predict_proba(trans_test)
            preds = (model1.predict_proba(trans_test) * .50 + model2.predict_proba(trans_test) * .50)
            results[output].append(log_loss(test_test, preds))
            print "elapsed "
            print time.time() - t0
        break


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
    #train = sample_data_random(df, 0.25)
    train = df[0:int(df.shape[0] * .30)]
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

    #train = sample_data_random(df, 0.75)
    train = df[0:int(df.shape[0] * .75)]
    labels_train = train[outputs]

    #test = sample_data_random(df, 0.25)
    test = df[0:int(df.shape[0] * .25)]
    labels_test = test[outputs]

    models = dict()

    for output in outputs:
        t0 = time.time()

        pipe_clf = CustomPipeline.get_pipeline()
        print "Fitting %s" % (output,)
        pipe_clf.fit(train, labels_train[output])
        preds = pipe_clf.predict_proba(test)
        results[output] = log_loss(labels_test[output], preds)
        models[output] = pipe_clf

        print "elapsed "
        print time.time() - t0

    sum_result = 0
    for result in results:
        print "Results %s: " % (result,) + str(results[result])
        sum_result += results[result]
    print 1.0 * sum_result / len(results)
    f = open("test_" + str(datetime.datetime.now()), 'wt')
    f.write(str(1.0 * sum_result / len(results)))
    return models


train = pd.read_csv('TrainingData.csv')
train = train.drop_duplicates(subset=text_features+outputs).reset_index()
for num_feature in num_features:
    train[num_feature] = train[num_feature].fillna(train[num_feature].mean())

test = pd.read_csv('TestData.csv')
for num_feature in num_features:
    test[num_feature] = test[num_feature].fillna(test[num_feature].mean())


validate_model(train)
models = test_model(train)

for output in outputs:
    print "predicting %s" % (output,)
    result = models[output].predict_proba(test)
    classes_ = models[output].steps[2][1].classes_
    sample[[output + '__' + entry for entry in classes_]] = result
    sample.to_csv("resultmixed.csv", index=False)

sample.to_csv("resultmixed.csv", index=False)
