# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:10:27 2018

@author: zsl
"""
import re
list=[[1,2,3],[4,5,6]]
list2=[1,3,4,5]
list3=['http://t.co/5HV6Sd8uQY','dfasdfadf']
list4='http://t.co/5HV6Sd8uQY'
str1='#bbctttttttt dadfa#bbcqttttttttt #bedroomtax  #disabilitya'
splits=str1.split()
for split in splits:  
    matchs=re.fullmatch(r'(#[a-z]{8,})',split)
    if matchs:            
        print(matchs)
        
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
word=wordnet_lemmatizer.lemmatize('scotland')
words = nltk.corpus.words.words()
word1='Scotland'
if word1 in words:
    print('yes')
def max_match(text):
    text=re.sub('[^A-Za-z]','',text)
    result=[]
    pos=0
    while len(text)>0:
        word=wordnet_lemmatizer.lemmatize(text[pos:len(text)])
        if word:
            if word in words:
                result.append(text[pos:len(text)])
                text=text[0:pos]
                pos=0
            elif word.capitalize() in words:
                print(yes)
            else:
                pos=pos+1
    return result
print(max_match(word))