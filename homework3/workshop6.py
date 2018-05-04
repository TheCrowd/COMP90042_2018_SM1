# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:28:26 2018

@author: zsl
"""

import numpy as np
tags = NNP, MD, VB, JJ, NN, RB, DT = 0, 1, 2, 3, 4, 5, 6
tag_dict = {0: 'NNP',
           1: 'MD',
           2: 'VB',
           3: 'JJ',
           4: 'NN',
           5: 'RB',
           6: 'DT'}
words = Janet, will, back, the, bill = 0, 1, 2, 3, 4

A = np.array([
    [0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025],
    [0.0008, 0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041],
    [0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231],
    [0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036],
    [0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068],
    [0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479],
    [0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]
    ])

pi = np.array([0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026])

B = np.array([
    [0.000032, 0, 0, 0.000048, 0],
    [0, 0.308431, 0, 0, 0],
    [0, 0.000028, 0.000672, 0, 0.000028],
    [0, 0, 0.000340, 0.000097, 0],
    [0, 0.000200, 0.000223, 0.000006, 0.002337],
    [0, 0, 0.010446, 0, 0],
    [0, 0, 0, 0.506099, 0]
    ])

alpha = np.zeros((len(tags), len(words))) # states x time steps
alpha[:,:] = float('-inf')
backpointers = np.zeros((len(tags), len(words)), 'int')

# base case, time step 0
alpha[:, 0] = pi * B[:,Janet]
np.set_printoptions(precision=2)
print (alpha)
# time step 1
for t1 in tags:
    for t0 in tags:
        score = alpha[t0, 0] * A[t0, t1] * B[t1, will]
        if score > alpha[t1, 1]:
            alpha[t1, 1] = score
            backpointers[t1, 1] = t0
print (alpha)

# time step 2
for t2 in tags:
    for t1 in tags:
        score = alpha[t1, 1] * A[t1, t2] * B[t2, back]
        if score > alpha[t2, 2]:
            alpha[t2, 2] = score
            backpointers[t2, 2] = t1
print (alpha)

# time step 3
for t3 in tags:
    for t2 in tags:
        score = alpha[t2, 2] * A[t2, t3] * B[t3, the]
        if score > alpha[t3, 3]:
            alpha[t3, 3] = score
            backpointers[t3, 3] = t2
print (alpha)

# time step 4
for t4 in tags:
    for t3 in tags:
        score = alpha[t3, 3] * A[t3, t4] * B[t4, bill]
        if score > alpha[t4, 4]:
            alpha[t4, 4] = score
            backpointers[t4, 4] = t3
print (alpha)

t4 = np.argmax(alpha[:, 4])
print (tag_dict[t4])

t3 = backpointers[t4, 4]

t2 = backpointers[t3, 3]
print (tag_dict[t2])
t1 = backpointers[t2, 2]
print (tag_dict[t1])
t0 = backpointers[t1, 1]
print (tag_dict[t0])
print (tag_dict[t3])

def viterbi(params, words):
    pi, A, B = params
    N = len(words)
    T = pi.shape[0]
    
    alpha = np.zeros((T, N))
    alpha[:, :] = float('-inf')
    backpointers = np.zeros((T, N), 'int')
    
    # base case
    alpha[:, 0] = pi * B[:, words[0]]
    
    # recursive case
    for w in range(1, N):
        for t2 in range(T):
            for t1 in range(T):
                score = alpha[t1, w-1] * A[t1, t2] * B[t2, words[w]]
                if score > alpha[t2, w]:
                    alpha[t2, w] = score
                    backpointers[t2, w] = t1
    
    # now follow backpointers to resolve the state sequence
    output = []
    output.append(np.argmax(alpha[:, N-1]))
    for i in range(N-1, 0, -1):
        output.append(backpointers[output[-1], i])
    
    return list(reversed(output)), np.max(alpha[:, N-1])

output, score = viterbi((pi, A, B), [Janet, will, back, the, bill])
print ([tag_dict[o] for o in output])
print (score)
output, score = viterbi((pi, A, B), [Janet, will, back, the, Janet, back, bill])
print ([tag_dict[o] for o in output])
print (score)


#supervised training
from nltk.corpus import treebank
corpus = treebank.tagged_sents()
print(corpus)
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
    
word_names = [None] * len(word_numbers)
for word, index in word_numbers.items():
    word_names[index] = word
tag_names = [None] * len(tag_numbers)
for tag, index in tag_numbers.items():
    tag_names[index] = tag

training = num_corpus[:-10] # reserve the last 10 sentences for testing
testing = num_corpus[-10:]

S = len(tag_numbers)
V = len(word_numbers)

# initalise
eps = 0.1
pi = eps * np.ones(S)
A = eps * np.ones((S, S))
B = eps * np.ones((S, V))

# count
for sent in training:
    last_tag = None
    for word, tag in sent:
        B[tag, word] += 1
        # bug fixed here 27/3/17; test was incorrect 
        if last_tag == None:
            pi[tag] += 1
        else:
            A[last_tag, tag] += 1
        last_tag = tag
        
# normalise
pi /= np.sum(pi)
for s in range(S):
    B[s,:] /= np.sum(B[s,:])
    A[s,:] /= np.sum(A[s,:])
predicted, score = viterbi((pi, A, B), list(map(lambda w_t: w_t[0], testing[0])))

print('%20s\t%5s\t%5s' % ('TOKEN', 'TRUE', 'PRED'))
for (wi, ti), pi in zip(testing[0], predicted):
    print('%20s\t%5s\t%5s' % (word_names[wi], tag_names[ti], tag_names[pi]))