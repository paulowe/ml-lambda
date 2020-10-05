#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.externals import joblib


"""
Load Data and split training/testing
"""

data = pd.read_csv('data/syntheticData.csv')
X, y = data.iloc[:, 1:], data.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)

"""
Now use scikit-learn's MLP classifier to train the model.
"""
clf = MLPClassifier()
clf = clf.fit(X_train, y_train)

"""
Serialize Model
"""
joblib.dump(clf, 'model/mlp.pkl')
