# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:18:32 2018

@author: zsl
"""
## code copied from wsta_n6_hidden_markov_models ########
from nltk.corpus import treebank
corpus = treebank.tagged_sents()
word_numbers = {}
tag_numbers = {}

num_corpus = []
for sent in corpus:
    num_sent = []
    for word, tag in sent:
        wi = word_numbers.setdefault(word.lower(), len(word_numbers))
        ti = tag_numbers.setdefault(tag, len(tag_numbers))
        num_sent.append((wi, ti))
    num_corpus.append(num_sent)   
#copy end
    
# print first preprocessed sentence
print (num_corpus[0])
# print the indice for word 'electricity'
print(word_numbers.get('electricity',-1))
#print the length of full tagset
print(len(tag_numbers))

from nltk.corpus import twitter_samples
import re
tweet_corpus=twitter_samples.tokenized()
# replace special tokens with special symbols

def word_preprocess(word):
    um_pattern=re.compile(r'(@[a-z0-9]*)')
    ht_pattern=re.compile(r'(#[a-z0-9]*)')
    RT_pattern=re.compile(r'(rt)')
    url_pattern=re.compile(r'(http://|https://)')
    word=word.lower()
    if(um_pattern.match(word)):
        word='USER_TOKEN'
    elif(ht_pattern.match(word)):
        word='HASHTAG_TOKEN'
    elif(RT_pattern.fullmatch(word)):
        word='RETWEET_TOKEN'
    elif(url_pattern.match(word)):
        word="URL_TOKEN"      
    return word

processed_corpus=[]
for tweet in tweet_corpus:
    tweet_list=[]
    for token in tweet:
        token=word_preprocess(token)
        wi = word_numbers.setdefault(token, len(word_numbers))
        tweet_list.append(wi)
    processed_corpus.append(tweet_list)

# print first preprocessed sentence
print (processed_corpus[0])
# print the indice for word 'electricity'
print(word_numbers.get('electricity',-1))               
# print the indice for word 'HASHTAG_TOKEN'
print(word_numbers.get('HASHTAG_TOKEN',-1)) 

extra_word='<unk>'
word_numbers.setdefault(extra_word, len(word_numbers))

extra_tags=['USR','HT','RT','URL','VPP','TD','O']
for tag in extra_tags:
    tag_numbers.setdefault(tag, len(tag_numbers))

word_names = [None] * len(word_numbers)
for word, index in word_numbers.items():
    word_names[index] = word
tag_names = [None] * len(tag_numbers)
for tag, index in tag_numbers.items():
    tag_names[index] = tag
# print the indice for '<unk>'
print(word_numbers.get('<unk>',-1))
#print the length of full tagset
print(len(tag_numbers))

import urllib
try:
    urllib.request.urlretrieve("https://github.com/aritter/twitter_nlp/raw/master/data/annotated/pos.txt","pos.txt")
except: # Python 2
    urllib.urlretrieve("https://github.com/aritter/twitter_nlp/raw/master/data/annotated/pos.txt","pos.txt")
def pos_preprocess(pos):
    if pos=="(":
        pos='-LRB-'
    elif pos==")":
        pos='-RRB-'
    elif pos=='NONE':
        pos='-NONE-'
    return pos
test_num_corpus=[] #store (word,tag) indice of corpus
test_word_numbers=[] #store only the word indices of corpus
test_tag_numbers=[] #store only the tag indices of corpus
with open('pos.txt') as f:
    test_sent_num=[] 
    word_num=[]
    tag_num=[]
    for line in f:
        if line.strip() == '':
            test_num_corpus.append(test_sent_num)
            test_word_numbers.append(word_num)
            test_tag_numbers.append(tag_num)
            test_sent_num=[]
            word_num=[]
            tag_num=[]
        else:
            word, pos = line.strip().split()
            word=word_preprocess(word)
            pos=pos_preprocess(pos)
            if word in word_numbers:
                test_sent_num.append((word_numbers[word],tag_numbers[pos]))
                word_num.append(word_numbers[word])
                tag_num.append(tag_numbers[pos])
            else:
                test_sent_num.append((word_numbers['<unk>'],tag_numbers[pos]))
                word_num.append(word_numbers['<unk>'])
                tag_num.append(tag_numbers[pos])
print('##### first sentence of test corpus  ############')
print (test_num_corpus[0])

import numpy as np
## code copied from wsta_n6_hidden_markov_models ########
def count(tagged_corpus,words_dict,tags_dict):
    S = len(tags_dict)
    V = len(words_dict)

    # initalise
    eps = 0.1
    pi = eps * np.ones(S)
    A = eps * np.ones((S, S))
    O = eps * np.ones((S, V))

    # count
    for sent in tagged_corpus:
        last_tag = None
        for word, tag in sent:
            if(tag=='O'):
                print('bingo')
            O[tag, word] += 1
            if last_tag == None:
                pi[tag] += 1
            else:
                A[last_tag, tag] += 1
            last_tag = tag
        
    # normalise
    pi /= np.sum(pi)
    for s in range(S):
        O[s,:] /= np.sum(O[s,:])
        A[s,:] /= np.sum(A[s,:])
    return pi,A,O
## copy end ########
pi,A,O=count(num_corpus,word_numbers,tag_numbers)   

## code copied from wsta_n6_hidden_markov_models ########  
def viterbi(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf')
    backpointers = np.zeros((M, S), 'int')
    
    # base case
    alpha[0, :] = pi * O[:,observations[0]]
    
    # recursive case
    for t in range(1, M):
        for s2 in range(S):
            for s1 in range(S):
                score = alpha[t-1, s1] * A[s1, s2] * O[s2, observations[t]]
                if score > alpha[t, s2]:
                    alpha[t, s2] = score
                    backpointers[t, s2] = s1
    
    # now follow backpointers to resolve the state sequence
    ss = []
    ss.append(np.argmax(alpha[M-1,:]))
    for i in range(M-1, 0, -1):
        ss.append(backpointers[i, ss[-1]])
    ss=reversed(ss)
    result=[]
    for word,tag in zip(observations,ss):
        result.append((word,tag))
    return result
#copy end##########
predicted=[]
for sent in test_word_numbers:
    predicted.append(viterbi((pi,A,O), sent))
print()
print('##### first sentence of predicted list ############')
print(predicted[0])

def get_tag_list(sent):
    tags=[]
    for word,tag in sent:
        tags.append(tag_names[tag])
    return tags

true_tags=[]
for sent in test_num_corpus:
    true_tags.append(get_tag_list(sent))

pred_tags=[]
for sent in predicted:
    pred_tags.append(get_tag_list(sent))
    
from sklearn.metrics import accuracy_score as acc

# flat our data into single list
all_test_tags = [tag for tags in true_tags for tag in tags]
# flat predicted data into single list
all_pred_tags = [tag for tags in pred_tags for tag in tags]
print('##########prediction accuracy#######')
print (acc(all_test_tags, all_pred_tags))

#adapt emision matrix
wordIndex=[word_numbers['USER_TOKEN'],word_numbers['HASHTAG_TOKEN'],word_numbers['RETWEET_TOKEN'],word_numbers['URL_TOKEN']] 
tagIndex=[tag_numbers['USR'],tag_numbers['HT'],tag_numbers['RT'],tag_numbers['URL']]   
adapted_O=O.copy()
for word,tag in zip(wordIndex,tagIndex):
    adapted_O[tag,:]=0
    adapted_O[tag,word]=1
print(adapted_O)

#evaluate new tagger on test tweet again
new_pred=[]
for sent in test_word_numbers:
    new_pred.append(viterbi((pi,A,adapted_O), sent))

new_pred_tags=[]
for sent in new_pred:
    new_pred_tags.append(get_tag_list(sent))
    
from sklearn.metrics import accuracy_score as acc


# flat predicted data into single list
new_all_pred_tags = [tag for tags in new_pred_tags for tag in tags]
 
from sklearn.metrics import f1_score,classification_report
result_array=f1_score(all_test_tags, new_all_pred_tags,labels=tag_names,average=None)
Fscores=zip(tag_names,result_array)
Fscores=sorted(Fscores, key=lambda x:x[1])
for tag,fscore in Fscores:
    print (tag,": ",fscore)
print(classification_report(all_test_tags, new_all_pred_tags))
print('##########new prediction accuracy#######')
print (acc(all_test_tags, new_all_pred_tags))
print("####tagger performs worst on the follow tags:")
for tag,fscore in Fscores:
    if fscore==0.0:
        print (tag)
print("####tagger performs best on the tag:")
print(Fscores[-1][0])

print(set(all_test_tags)-set(all_pred_tags))
print(set(all_test_tags)-set(new_all_pred_tags))

##extra credits
def log_count(tagged_corpus,words_dict,tags_dict):
    S = len(tags_dict)
    V = len(words_dict)

    # initalise
    eps = 0.1
    pi = eps * np.ones(S)
    A = eps * np.ones((S, S))
    O = eps * np.ones((S, V))

    # count
    for sent in tagged_corpus:
        last_tag = None
        for word, tag in sent:
            O[tag, word] += 1
            if last_tag == None:
                pi[tag] += 1
            else:
                A[last_tag, tag] += 1
            last_tag = tag
        
    # normalise
    pi /= np.sum(pi)
    for s in range(S):
        O[s,:] /= np.sum(O[s,:])
        A[s,:] /= np.sum(A[s,:])
    return np.log(pi),np.log(A),np.log(O)

import operator
def log_viterbi(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf')
    backpointers = np.zeros((M, S), 'int')
    
    # base case
    alpha[0, :] = pi + O[:,observations[0]]
    
    # recursive case
    for t in range(1, M):
        for s in range(S):
                score = alpha[t-1, :] + A[:, s]
                backpointers[t,s],alpha[t,s]=max(enumerate(score),key=operator.itemgetter(1))
                alpha[t,s]=alpha[t,s]+O[s, observations[t]]
                  
    
    # now follow backpointers to resolve the state sequence
    ss = []
    ss.append(np.argmax(alpha[M-1,:]))
    for i in range(M-1, 0, -1):
        ss.append(backpointers[i, ss[-1]])
    ss=reversed(ss)
    result=[]
    for word,tag in zip(observations,ss):
        result.append((word,tag))
    return result
#print(log_viterbi((np.log(pi),np.log(A),np.log(O)),test_word_numbers[0]))
#print(test_num_corpus[0])

#Hard EM
#initial params

EM_pi=np.log(pi)
EM_A=np.log(A)
EM_O=np.log(adapted_O)
#tag training tweet
tweet_train_num=[]
for sent in processed_corpus:
    tweet_train_num.append(log_viterbi((EM_pi,EM_A,EM_O),sent))
for i in range(5):
    all_train_num=tweet_train_num+num_corpus
    EM_pi,EM_A,EM_O=log_count(all_train_num,word_numbers,tag_numbers)
    tweet_pred_num=[]
    for sent in processed_corpus:
        tweet_pred_num.append(log_viterbi((EM_pi,EM_A,EM_O),sent))
    
    tweet_real_tags=[]
    for sent in tweet_train_num:
        tweet_real_tags.append(get_tag_list(sent))

    tweet_pred_tags=[]
    for sent in tweet_pred_num:
        tweet_pred_tags.append(get_tag_list(sent))
    print("###prediction accuracy EM iteration ", i )
          
    # flat our data into single list
    tweet_real_taglist = [tag for tags in tweet_real_tags for tag in tags]
    # flat predicted data into single list
    tweet_pred_taglist = [tag for tags in tweet_pred_tags for tag in tags]
    print(acc(tweet_real_taglist, tweet_pred_taglist))
    tweet_train_num=tweet_pred_num

#soft EM    
from scipy.misc import logsumexp
def forward(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf')
    
    # base case
    alpha[0, :] = pi + O[:,observations[0]]
    
    # recursive case
    for t in range(1, M):
        for s in range(S):
                score = alpha[t-1, :] + A[:, s]
                alpha[t,s]=logsumexp(score)
                alpha[t,s]=alpha[t,s]+O[s, observations[t]]
                
    result=logsumexp(alpha[M-1,:])
    return alpha,result
 
def backward(params,observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    beta = np.zeros((M, S))
    beta[:,:] = float('-inf')
    
    # base case
    beta[M-1, :] = 0
    
    # recursive case
    for t in range(M-2,-1,-1):
        for s in range(S):
                score = beta[t+1, :] + A[s,:]+O[:, observations[t+1]]
                beta[t,s]=logsumexp(score)
                
    result=logsumexp(pi+beta[0,:]+O[:, observations[0]])
    return beta,result

#alpha1,result1=forward((np.log(pi),np.log(A),np.log(O)),test_word_numbers[0])
#beta1,result2=backward((np.log(pi),np.log(A),np.log(O)),test_word_numbers[0])
#print(result1)
#print(result2)

def expected_count(alpha,beta,magrin_a,margin_b,words_dict,tags_dict):
   
    S = len(tags_dict)
    V = len(words_dict)

    # initalise
    eps = 0.1
    A = eps * np.ones((S, S))
    O = eps * np.ones((S, V))
     #update emisstion possbilities
     for t in range(S):
         for w in range(V):
             
    