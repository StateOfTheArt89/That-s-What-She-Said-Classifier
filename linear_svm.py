from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib

import json
import random
import numpy as np

with open('data/all_samples.json') as data_file:
    data_json = json.load(data_file)
    random.shuffle(data_json)

train_json, test_json = train_test_split(data_json, train_size = 0.8)

data_text_train = []
data_label_train = []

for sample in train_json:
    data_text_train.append(sample['text'])
    if sample['is_twss']:
        data_label_train.append(1)
    else:
        data_label_train.append(0)

data_text_test = []
data_label_test = []

for sample in test_json:
    data_text_test.append(sample['text'])
    if sample['is_twss']:
        data_label_test.append(1)
    else:
        data_label_test.append(0)


text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), min_df=1)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l1',
                                            alpha=1e-4, n_iter=50, random_state=1337)),])

_ = text_clf.fit(data_text_train, data_label_train)

predicted = text_clf.predict(data_text_train)
print "Train accuracy: %f" % np.mean(predicted == data_label_train)

predicted = text_clf.predict(data_text_test)
print "Test accuracy: %f" % np.mean(predicted == data_label_test)


joblib.dump(text_clf, 'model/text_cfl.pkl')
