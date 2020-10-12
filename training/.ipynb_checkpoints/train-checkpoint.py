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


'''
A. Introduce CV set

B. Plot learning curves / Evaluation tactics

#1. Cost/Error - Lambda (Jcv , Jtrain)
#2. Cost/Error - #layers (Jcv, Jtrain)
#3. Cost/Error - # training examples (Jcv, Jtrain)
#4. AUC, Accuracy, Precicision, Recall ...

C. Potential action plans

 - Get more training examples (fixes overfitting)
 - Try smaller set of features (fixes overfitting)
 - Try getting additional features (fixes underfitting)
 - Add polynomial features (underfitting)
 - decreasing lambda (fix underfitting)
 - increasing lambda (fixes overfitting)

'''


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
