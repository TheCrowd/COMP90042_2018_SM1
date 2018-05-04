# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 22:18:16 2018

@author: zsl
"""

from nltk.corpus import wordnet as wn
def lemma_count(word,synsets):
    counts={}
    for synset in synsets:
        count=0
        for lemma in synset.lemmas():
            if word ==lemma.name():
                count=count+lemma.count()
        counts[synset.name()]=counts.get(synset.name(),0)+count
    counts=sorted(counts.items(),key=operator.itemgetter(1),reverse= True)
    return counts[0],counts[1]

def isValid(word):
    dictionary={}
    words = wn.synsets(word)
    
    if (len(words) == 1) and ('.n.' in  str(words[0])):
        dictionary[word] = words
        return [True, dictionary]
    else: 
        list_1 = []
        dict_1={}
        for synset in words:
            for lemma in synset.lemmas():
                if word == lemma.name():
                    dict_1[synset] = lemma.count() 
#                print synset, dict[synset]
        for (key, value) in sorted(dict_1.iteritems(), key = lambda (k,v):(v,k), reverse = True):
            list_1.append([key, value])
        print (list_1)
        if ('.n.' in str(words[0])) and list_1[0][1] >= 5 and list_1[0][1] >= 5 * list_1[1][1]:
            dictionary[word] = words[0]
            return [True, dictionary]
    return [False, None]
#print(isValid('monk'))
synsets=wn.synsets('monk')
print(lemma_count('monk',synsets))