# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:43:49 2018

@author: zsl
"""

from nltk.corpus import wordnet as wn
dogs=wn.synsets('dog')
lemma_str=[]
for synset in dogs:
    print(synset)
    for lemma in synset.lemmas():
        lemma_str.append(str(lemma))
#        print(str(lemma))
        print(lemma)
#set1=set(['professor','doctor'])
#set2=set(['professor','doctor','1','2'])
#print(set1&set2)       