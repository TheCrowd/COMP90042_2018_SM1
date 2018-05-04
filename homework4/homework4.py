# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 15:03:27 2018

@author: zsl
"""
from __future__ import print_function 

def hangman(secret_word, guesser, max_mistakes=8, verbose=True, **guesser_args):
    """
        secret_word: a string of lower-case alphabetic characters, i.e., the answer to the game
        guesser: a function which guesses the next character at each stage in the game
            The function takes a:
                mask: what is known of the word, as a string with _ denoting an unknown character
                guessed: the set of characters which already been guessed in the game
                guesser_args: additional (optional) keyword arguments, i.e., name=value
        max_mistakes: limit on length of game, in terms of allowed mistakes
        verbose: be chatty vs silent
        guesser_args: keyword arguments to pass directly to the guesser function
    """
    secret_word = secret_word.lower()
    mask = ['_'] * len(secret_word)
    guessed = set()
    if verbose:
        print("Starting hangman game. Target is", ' '.join(mask), 'length', len(secret_word))
    
    mistakes = 0
    while mistakes < max_mistakes:
        if verbose:
            print("You have", (max_mistakes-mistakes), "attempts remaining.")
        guess = guesser(mask, guessed, **guesser_args)

        if verbose:
            print('Guess is', guess)
        if guess in guessed:
            if verbose:
                print('Already guessed this before.')
            mistakes += 1
        else:
            guessed.add(guess)
            if guess in secret_word:
                for i, c in enumerate(secret_word):
                    if c == guess:
                        mask[i] = c
                if verbose:
                    print('Good guess:', ' '.join(mask))
            else:
                if verbose:
                    print('Sorry, try again.')
                mistakes += 1
                
        if '_' not in mask:
            if verbose:
                print('Congratulations, you won.')
            return mistakes
        
    if verbose:
        print('Out of guesses. The word was', secret_word)    
    return mistakes

def human(mask, guessed, **kwargs):
    """
    simple function for manual play
    """
    print('Enter your guess:')
    try:
        return raw_input().lower().strip() # python 3
    except NameError:
        return input().lower().strip() # python 2
    
from nltk.corpus import brown
import numpy as np
import re,string
brown_words=brown.words();
words=set()
pattern=re.compile(r'^[a-z]+$')
for word in brown_words:
    word=word.lower()
    if re.match(pattern,word):
      words.add(word)  
word_types=list(words)
np.random.shuffle(word_types)
train_set=word_types[1000:]
test_set=word_types[:1000]
print(len(word_types))
print(len(train_set))
print(len(test_set))

#####################################
def trivial_guess(mask, guessed, **kwargs):
    alphas=list(string.ascii_lowercase)
    for word in guessed:
        if word in alphas:
            alphas.remove(word)
    return np.random.choice(alphas)

def average_mistakes(data,guesser,**kwargs):
    total_mistakes=0
    for word in data:
        mistakes=hangman(word,guesser, max_mistakes=26, verbose=False,**kwargs)
        total_mistakes+=mistakes
    return total_mistakes/len(data)
#print average mistakes of trivial guesser
print(average_mistakes(test_set,trivial_guess))

#count char frequency in train set
import operator
char_freq={}
total_char=0
for word in train_set:
    total_char=total_char+len(word)
    for char in word:
        char_freq[char]=char_freq.get(char,0)+1
for key in char_freq.keys():
    char_freq[key]=char_freq.get(key,0)/total_char
char_freq_list=sorted(char_freq.items(),key=operator.itemgetter(1),reverse=True)

def unigram_guess(mask, guessed, **kwargs):
    char_prob=kwargs['char_freq']
    for item in char_prob:
        if item[0] not in guessed:
            return item[0]
    return None
#print average mistakes of unigram guesser
print(average_mistakes(test_set,unigram_guess,char_freq=char_freq_list))

len_char_freq={}
len_total_char={}
for word in train_set:
    word_len=len(word)
    len_total_char[word_len]=len_total_char.get(word_len,0)+word_len
    char_freq=len_char_freq.get(word_len,{})
    for char in word:
        char_freq[char]=char_freq.get(char,0)+1
    len_char_freq[word_len]=char_freq
    
for k,v in len_char_freq.items():
    if(len_char_freq[k].keys()!=26):
        for char in list(string.ascii_lowercase):
                len_char_freq[k][char]=len_char_freq[k].get(char,0)+1
                len_total_char[k]=len_total_char.get(k,0)+1

for k,v in len_char_freq.items():
    for key,value in v.items():
        v[key]=value/len_total_char[k]

for key in len_char_freq.keys():    
    len_char_freq[key]=sorted(len_char_freq[key].items(),key=operator.itemgetter(1),reverse=True)
len_char_freq[0]=char_freq_list


def advanced_unigram(mask, guessed, **kwargs):
    word_len=len(mask)
    len_char_freq=kwargs['len_char_freq']
    if word_len in len_char_freq.keys():
        for item in len_char_freq[word_len]:
            if item[0] not in guessed:
                return item[0]
    else:
        for item in len_char_freq[0]:
            if item[0] not in guessed:
                return item[0]
#print average mistakes of advanced unigram guesser
print(average_mistakes(test_set,advanced_unigram,len_char_freq=len_char_freq))

#padding for words in ngram model,use symbol '<' at the start, and pad the end with '>' 
def word_padding(word,n):
    paddingL=[]
    paddingR=[]
    for i in range(n-1):
        paddingL.append('<')
    for i in range(n-1):
        paddingR.append('>')
    
    return ''.join(paddingL)+word+''.join(paddingR)



def get_count(words,n):
    count={}
    for word in words:
        word=word_padding(word,5)
        if len(word)>=n:
            for i in range(len(word)-n+1):
                keyStr="".join(word[i:i+n])
                count[keyStr]=count.get(keyStr,0)+1
    return count

import math

def get_total_log_prob(word,char,counts,lambdas, token_count):
    blank_count=word.count('_')
    cur_pos=0       
    blank_probs=[]
    for i in range(blank_count):
        index=word[cur_pos:].find('_')
        if index!=-1:
            start_index=index
            while word[start_index-1]!='_':
                start_index=start_index-1
            log_prob=get_log_prob_interp(word[start_index:index]+char,counts[0:index-start_index+1],lambdas,token_count)
            blank_probs.append(log_prob)
        cur_pos=index+1
    return sum(blank_probs)
    
def get_log_prob_interp(word,counts,lambdas,token_count):
    degree=len(counts)
    probs=[]
    for i in range(degree):
        if i<degree-1:
            if counts[degree-i-1].get(word[i:],0)!=0:
                prob=counts[degree-i-1].get(word[i:],0)/counts[degree-i-2].get(word[i:-1],0)*lambdas[degree-i-1]
                probs.append(prob)
        elif i==degree-1:
            prob=counts[0].get(word[-1],0)/token_count*lambdas[0]
            probs.append(prob)
    return math.log(sum(probs))

def find_most_context(word,n):
    blanks=[]
    for i in range(n):
        blanks.append('_')
    cur_context=''.join(blanks)
    for i in range(0,len(word)-n):
        if word[i:i+n].count('_')<cur_context.count('_') and word[i+n-1] is '_':
            cur_context=word[i:i+n];
    return cur_context

#**kwargs   n = ngram degree, counts= count for every gram model,token_count=token_count
def ngram_guess(mask,guessed,**kwargs):
    n=kwargs['n']
    counts=kwargs['counts']
    token_count=kwargs['token_count']
    lambdas=kwargs['lambdas']
    maskStr=word_padding(''.join(mask),n)
    gram_item=find_most_context(maskStr,n)
    alphas=list(string.ascii_lowercase)
    for word in guessed:
        if word in alphas:
            alphas.remove(word)
    prob=float('-inf')
    maxChar=np.random.choice(list(alphas))      
    for char in alphas:
#        gram_item=gram_item[:-1]+char
        temp_prob=get_total_log_prob(gram_item,char,counts,lambdas,token_count)
        if temp_prob>prob:
            prob=temp_prob
            maxChar=char
    return maxChar
#get train data
unigram_count=get_count(train_set,1)
bigram_count=get_count(train_set,2)
trigram_count=get_count(train_set,3)
quagram_count=get_count(train_set,4)
pentagram_count=get_count(train_set,5)
token_count=sum(unigram_count.values())
counts=[unigram_count,bigram_count,trigram_count,quagram_count,pentagram_count]
lambdas=[0.01,0.03,0.06,0.3,0.6]    
#print average mistakes of ngram guesser
print(average_mistakes(test_set,ngram_guess,n=3,counts=counts[0:3],token_count=token_count,lambdas=lambdas))
print(average_mistakes(test_set,ngram_guess,n=4,counts=counts[0:4],token_count=token_count,lambdas=lambdas))
print(average_mistakes(test_set,ngram_guess,n=5,counts=counts[0:5],token_count=token_count,lambdas=lambdas))

#print(hangman('apple',ngram_guess, max_mistakes=16,n=3,counts=counts[0:3],token_count=token_count,lambdas=lambdas)) 
#print(hangman('apple',ngram_guess, max_mistakes=16,n=4,counts=counts[0:4],token_count=token_count,lambdas=lambdas)) 
#print(hangman('scout',ngram_guess, max_mistakes=16,n=5,counts=counts,token_count=token_count,lambdas=lambdas)) 

import numpy as np
#def find_best_args(dev_set,n):
#    mistakes=26
#    best_args=[]
#    for i in range(10):
#        randoms=np.random.rand(n)
#        randoms_sum=sum(randoms)
#        randoms=randoms/randoms_sum
#        temp_mistakes=average_mistakes(dev_set,ngram_guess,n=n,counts=counts[0:n],token_count=token_count,lambdas=randoms)
#        if temp_mistakes<mistakes:
#            mistakes=temp_mistakes
#            best_args=randoms
#    return best_args
#dev_set=train_set[:1000]
#best_args=find_best_args(dev_set,3)
#print(average_mistakes(test_set,ngram_guess,n=3,counts=counts[0:3],token_count=token_count,lambdas=best_args))
#
#best_args=find_best_args(dev_set,4)
#print(average_mistakes(test_set,ngram_guess,n=4,counts=counts[0:4],token_count=token_count,lambdas=best_args))
#
#best_args=find_best_args(dev_set,5)
#print(average_mistakes(test_set,ngram_guess,n=5,counts=counts[0:5],token_count=token_count,lambdas=best_args))  