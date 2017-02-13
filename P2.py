# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:50:27 2016

@author: Piyush
@id: 1001244127
"""

import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.preprocessing import normalize

# Read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# Get the column names
original_headers = list(nba.columns.values)

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'

# The dataset contains attributes such as player name and team name. 
# We know that they are not useful for classification and thus do not 
# include them as features. 

feature_columns = ['MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','ORB','DRB','TRB','AST','BLK','TOV','PF','PS/G'] #62

# Pandas DataFrame allows you to select columns. 
# We use column selection to split the data into features and class. 
nba_features_tmp = nba[feature_columns]

nba_feature = normalize(nba_features_tmp, norm='l2')    #L2 normalization

nba_class = nba[class_column]

# Divide the test and train data using the normalized nba_features
train_feature, test_feature, train_class, test_class = \
    train_test_split(nba_feature, nba_class, stratify=nba_class, \
    train_size=0.75, test_size=0.25)

linearsvm = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, max_iter=10000).fit(train_feature, train_class)

print("Test set accuracy: {:.2f}".format(linearsvm.score(test_feature, test_class)))

# Confusion Matrix:
prediction = linearsvm.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))


## Cross Validation:
linearsvm = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, max_iter=10000)
scores = cross_val_score(linearsvm, nba_feature, nba_class, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))