# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 00:44:03 2018

@author: zsl
"""
#import math
import numpy as np
from scipy.special import logsumexp
A = np.array([[0.7,0.3],[0.4,0.6]])
O = np.array([[0.5,0.4,0.1],[0.1,0.3,0.6]])
pi=np.array([0.6,0.4])

def forward(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf')
    
    # base case
    alpha[0, :] = pi *O[:,observations[0]]
    
    # recursive case
    for t in range(1, M):
        for s in range(S):
                score = alpha[t-1, :] * A[:, s]
                alpha[t,s]=np.sum(score)
                alpha[t,s]=alpha[t,s]*O[s, observations[t]]
                
    result=np.sum(alpha[M-1,:])
    return alpha,result
def backward(params,observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    beta = np.zeros((M, S))
    beta[:,:] = float('-inf')
    
    # base case
    beta[M-1, :] = 1
    
    # recursive case
    for t in range(M-2,-1,-1):
        for s in range(S):
                score =beta[t+1, :]*A[s,:]*O[:, observations[t+1]]
                beta[t,s]=np.sum(score)             
    result=np.sum(pi*beta[0,:]*O[:, observations[0]])
    return beta,result
alpha1,result1=forward((pi,A,O),[0,1])
beta1,result2=backward((pi,A,O),[0,1])
print(result1)
print(result2)
