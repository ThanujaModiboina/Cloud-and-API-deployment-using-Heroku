# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 00:50:20 2023

@author: thanu
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

# Load the Iris dataset
iris_d = load_iris()
X = iris_d.data
y = iris_d.target

# Create an instance of the logistic regression classifier
clf = LogisticRegression(max_iter=10000)

# Train the dataset
clf.fit(X, y)

# Saving the model(serialization) 
pickle.dump(clf, open('model.pkl','wb'))

# Loading the model(deserialization) 
model = pickle.load(open('model.pkl','rb'))

