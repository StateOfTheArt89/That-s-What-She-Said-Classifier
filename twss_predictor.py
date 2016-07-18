from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib

import json
import random
import numpy as np
import fileinput

class TWSSPredictor:
    def __init__(self):
        self.text_clf = joblib.load('model/text_cfl.pkl')

    def predict(self, text):
        predicted = self.text_clf.predict([text])
        return predicted[0] == 1
