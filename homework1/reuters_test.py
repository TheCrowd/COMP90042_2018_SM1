# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:47:28 2018

@author: zsl
"""

import nltk
nltk.download("reuters") # if necessary
from nltk.corpus import reuters

from sklearn.feature_extraction import DictVectorizer

def get_BOW(text):
    BOW = {}
    for word in text:
        BOW[word] = BOW.get(word,0) + 1
    return BOW

def prepare_reuters_data(topic,feature_extractor):
    feature_matrix = []
    classifications = []
    for file_id in reuters.fileids():
        feature_dict = feature_extractor(reuters.words(file_id))   
        feature_matrix.append(feature_dict)
        if topic in reuters.categories(file_id):
            classifications.append(topic)
        else:
            classifications.append("not " + topic)
     
    vectorizer = DictVectorizer()
    dataset = vectorizer.fit_transform(feature_matrix)
    print(dataset.shape)
    return dataset,classifications

dataset,classifications = prepare_reuters_data("acq",get_BOW)