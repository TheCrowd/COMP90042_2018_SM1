# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:04:30 2018

@author: zsl
"""

#def norm_list(aList):
#    new_list=[]
#    for line in aList:
#        list_parts=[]
#        list_temp=list(line[0])
#        list_temp.sort()
#        list_parts.append(list_temp)
#        list_parts.append(line[1])
#        new_list.append(list_parts)
#    new_list=sorted(new_list, key=lambda x:x[0][0]) 
#    return new_list
#lista=[[['bcda','bcba'],1],[['efdfg','bcab'],1]]
#lista=norm_list(lista)
#word_similarity=[['1','2','3'],['2','3','5']]
#test_list=[]
#for word in word_similarity:
#    line_part=[word[0],word[1]]
#    line_parts=[line_part,word[2]]
#    test_list.append(line_parts)
#    
#def norm_list(aList):
#    new_list=[]
#    for line in aList:
#        list_parts=[]
#        list_temp=list(line[0])
#        list_temp.sort()
#        list_parts.append(list_temp)
#        list_parts.append(line[1])
#        new_list.append(list_parts)
#    new_list=sorted(new_list, key=lambda x:x[0][0])
#    final_list=[]
#    for line in new_list:
#        final_list.append(line[1])
#    return final_list
#test_list=norm_list(test_list)
import operator
from nltk.corpus import wordnet as wn
synsets=wn.synsets('egg')
for s in synsets:
    print(s)

def lemma_count(word,synsets):
    counts={}
    for synset in synsets:
        count=0
        for lemma in synset.lemmas():
            if word == lemma.name():
                count=count+lemma.count()
        counts[str(synset)]=counts.get(str(synset),0)+count
    counts=sorted(counts.items(),key=operator.itemgetter(1),reverse= True)
    return counts[0],counts[1]

def str_process(synset_str):
    synset_str=synset_str[8:len(synset_str)-2]
    return synset_str  

def single_sense(word):
    noun_str='.n.'
    synsets=wn.synsets(word)
    if len(synsets)==1 and noun_str in str(synsets[0]):
        return True,str_process(str(synsets[0]))
    most_count,second_count=lemma_count(word,synsets)
    if most_count[1]>=5 and most_count[1]>=5*second_count[1] and noun_str in most_count[0]:
        return True,str_process(most_count[0])
    return False,None
#print(single_sense('egg'))
print(single_sense('egg'))