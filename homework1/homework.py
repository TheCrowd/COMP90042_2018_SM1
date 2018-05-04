# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 01:31:52 2018

@author: zsl
"""

#q1
import nltk
nltk.download("twitter_samples")
from nltk.corpus import twitter_samples
raw=twitter_samples.raw()
tweets_str=twitter_samples.strings();
#print(tweets_str)
#print(len(raw))
count=0
tweet_num=len(tweets_str)
for tweet in tweets_str:
    count+=len(tweet)
average=count/tweet_num
print(average)

#q2
import re
hashtags=[]
pattern=re.compile(r'(#[a-z]{8,})')
for tweet in tweets_str:
    splits=tweet.split()
    for split in splits:
        match=pattern.fullmatch(split)
        if match:
            hashtags.append(match.group(0)) 
print(len(hashtags))
#print(hashtags)

#q3
words = nltk.corpus.words.words() # words is a Python list
from nltk.stem import WordNetLemmatizer

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma
def max_match(text):
    text=re.sub('[^A-Za-z]','',text)
    result=[]
    pos=0
    while len(text)>0:
        word=lemmatize(text[pos:len(text)])
        if word:
            if word in words:
                result.append(text[pos:len(text)])
                text=text[0:pos]
                pos=0
            else:
                pos=pos+1
    return result
tokenized_hashtag=[]
for hashtag in hashtags:
    temp=max_match(hashtag)
    temp.reverse()
    tokenized_hashtag.append(temp)
print(tokenized_hashtag[-20:])

#extra credit

def max_match_forward(text):
    text=re.sub('[^A-Za-z]','',text)
    result=[]
    pos=len(text)
    while len(text)>0:
        word=lemmatize(text[0:pos])
        if word:
            if word in words:
                result.append(text[0:pos])
                text=text[pos:]
                pos=len(text)
            else:
                pos=pos-1
    return result


tokenized_hashtag_forward=[]
for hashtag in hashtags:
    tokenized_hashtag_forward.append(max_match_forward(hashtag))
print(tokenized_hashtag_forward[-20:])

for i in range(len(tokenized_hashtag_forward)):
    if (len(tokenized_hashtag[i])==len(tokenized_hashtag_forward[i])):
        for j in range(len(tokenized_hashtag[i])):
            if tokenized_hashtag[i][j]!=tokenized_hashtag_forward[i][j]:
                print('reverse:')
                print(tokenized_hashtag[i])
                print('forward:')
                print(tokenized_hashtag_forward[i])
                print('suggested:')
                print(tokenized_hashtag[i])
                print('------------')
                break;
    else:
        print('reverse:')
        print(tokenized_hashtag[i])
        print('forward:')
        print(tokenized_hashtag_forward[i])
        print('suggeted:')
        if(len(tokenized_hashtag[i])<len(tokenized_hashtag_forward[i])):       
            print(tokenized_hashtag[i])
        else:
            print(tokenized_hashtag_forward[i])
        print('------------')
        

#text classification
positive_tweets = nltk.corpus.twitter_samples.tokenized("positive_tweets.json")
negative_tweets = nltk.corpus.twitter_samples.tokenized("negative_tweets.json")
for tweet in positive_tweets:
    tweet.append("positive")
for tweet in negative_tweets:
    tweet.append("negative")

all_tweets=positive_tweets+negative_tweets
import random
from nltk.corpus import stopwords

all_tweets=random.sample(all_tweets,len(all_tweets))
all_tweets_word=[]
for tweet in all_tweets:
    tweet_word=[]
    for token in tweet:
        if (token.isalpha() and 
        token not in stopwords.words("english") and
        re.match(r'[^a-zA-Z]',token) is None):
            tweet_word.append(token.lower())
    all_tweets_word.append(tweet_word)
#
#print(all_tweets_word[0:100])
train_set=all_tweets_word[0:int(len(all_tweets_word)*0.8)]
dev_set=all_tweets_word[int(len(all_tweets_word)*0.8):int(len(all_tweets_word)*0.9)]
test_set=all_tweets_word[int(len(all_tweets_word)*0.9):len(all_tweets_word)]

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

nb=MultinomialNB()
y_train=[]
for tweet in train_set:
    y_train.append(tweet[len(tweet)-1])
  

y_test=[]
for tweet in test_set:
   y_test.append(tweet[len(tweet)-1])

y_dev=[]
for tweet in dev_set:
    y_dev.append(tweet[len(tweet)-1])
    
no_label_train=[]
for tweet in train_set:
    no_label_train.append(tweet[0:len(tweet)-1])

no_label_test=[]
for tweet in test_set:
    no_label_test.append(tweet[0:len(tweet)-1])

no_label_dev=[]
for tweet in dev_set:
    no_label_dev.append(tweet[0:len(tweet)-1])

#code copied from wsta_n2_text_classification.ipynb    
def get_BOW(text):
    BOW = {}
    for word in text:
        BOW[word] = BOW.get(word,0) + 1
    return BOW

def prepare_data(tweets,feature_extractor):
    feature_matrix = []
    for tweet in tweets:
        feature_dict = feature_extractor(tweet)   
        feature_matrix.append(feature_dict)
  
    vectorizer = DictVectorizer()
    dataset = vectorizer.fit_transform(feature_matrix)
    feature_names=vectorizer.get_feature_names()
    return dataset,feature_names
#copy end

x_matrix,feature_names=prepare_data(no_label_train+no_label_test+no_label_dev,get_BOW)

x_train=x_matrix[0:int(len(all_tweets_word)*0.8)]
x_test=x_matrix[int(len(all_tweets_word)*0.8):int(len(all_tweets_word)*0.9)]
x_dev=x_matrix[int(len(all_tweets_word)*0.9):]
clf = MultinomialNB()


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
def tune_parameters(x,y,tuned_params,estimator):
    # Split the dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
    if type(estimator)==type(MultinomialNB()):
        scores=['precision', 'recall']
    else:
        scores=['recall']
    for score in scores: 
        print()
        print("# Tuning hyper-parameters for %s" % score)
       
        clf = GridSearchCV(estimator, tuned_params, cv=5,
                       scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
#        print()
        print(clf.best_params_)
        print()
        print("Grid scores on different settings:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        params = clf.cv_results_['params']
        for mean, std, params in zip(means, stds, params):
            print("mean:%f and std:%f for %r"% (mean, std * 2, params))
        print()

#        print("Detailed classification report:")
#        print()
#        y_true, y_pred = y_test, clf.predict(X_test)
#        print(classification_report(y_true, y_pred))
#        print()
    return clf.best_params_
tuned_params=[{'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]}]
best_alpha=tune_parameters(x_dev,y_dev,tuned_params,clf)
MultinomialNB(best_alpha, class_prior=None, fit_prior=True)

clf2=LogisticRegression()
tuned_params_reg=[{'C':[0.001,0.01,0.1,1,10,30,60,100]}]
best_C=tune_parameters(x_dev,y_dev,tuned_params_reg,clf2)
LogisticRegression(best_C)

#use Multinominal to predict test set
clf.fit(x_train, y_train)
y_predict_NB=clf.predict(x_test)
print('MultinominalNB:\n'+classification_report(y_test, y_predict_NB))

#use LogisticRegression to predict test set
clf2.fit(x_train,y_train)
y_predict_LR=clf2.predict(x_test)
print('LogisticRegression:\n'+classification_report(y_test, y_predict_LR))

