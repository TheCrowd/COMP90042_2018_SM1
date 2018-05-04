## -*- coding: utf-8 -*-
#"""
#Created on Mon Apr  9 15:23:43 2018
#
#@author: zsl
#"""
##import re
##um_pattern=re.compile(r'(@[a-z0-9]*)')
##hg_pattern=re.compile(r'(#[a-z0-9]*)')
##URL_pattern=re.compile(r'(http://|https://)')
##urlStr="http://www.baidu.com"
##urlStr1="https://www.baidu.com"
##htstr="#sf1231sdfsdf"
##umstr="@dfasdfl2123"
##print(URL_pattern.match(urlStr))
##print(URL_pattern.match(urlStr1))
##print(um_pattern.match(umstr))
##print(hg_pattern.match(htstr))
#def get_counts(words):
#    uni_count={}
#    bi_count={}
#    tri_count={}
#    qua_count={}
#    penta_count={}
#    for word in words:
#        for char in word:
#            uni_count[char]= uni_count.get(char,0)+1
#        for i in range(len(word)-1):
#            keyStr="".join(word[i:i+2])
#            bi_count[keyStr]=bi_count.get(keyStr,0)+1
#        for i in range(len(word)-2):
#            keyStr="".join(word[i:i+3])
#            tri_count[keyStr]=tri_count.get(keyStr,0)+1
#        for i in range(len(word)-3):
#            keyStr="".join(word[i:i+4])
#            qua_count[keyStr]=qua_count.get(keyStr,0)+1
#        for i in range(len(word)-4):
#            keyStr="".join(word[i:i+5])
#            penta_count[keyStr]=penta_count.get(keyStr,0)+1
#    return uni_count,bi_count,tri_count,qua_count,penta_count
#
#def get_count(words,n):
#    count={}
#    for word in words:
#        if len(word)>=n:
#            for i in range(len(word)-n+1):
#                keyStr="".join(word[i:i+n])
#                count[keyStr]=count.get(keyStr,0)+1
#    return count
#print(get_count(['apple'],1))
#
#def word_padding(word,n):
#    padding=[]
#    for i in range(n-1):
#        padding.append('>')
#    
#    return ''.join(padding)+word
#import math
#def get_log_prob_interp(word,counts,lambdas,token_count):
#    degree=len(counts)
#    probs=[]
#    for i in range(degree):
#        if i<degree-1:
#            prob=counts[degree-i-1]/counts[degree-i-2]*lambdas[degree-i-1]
#        elif i==degree-1:
#            prob=counts[0]/token_count*lambdas[0]
#        probs.append(prob)
#    return math.log(sum(probs))
#
#def find_most_context(word,n):
#    blanks=[]
#    for i in range(n):
#        blanks.append('-')
#    cur_context=''.join(blanks)
#    for i in range(0,len(word)-n):
#        if word[i:i+n].count('-')<cur_context.count('-') and word[i+n-1] is '-':
#            cur_context=word[i:i+n];
#    return cur_context
#print(get_log_prob_interp('apple',[0.5,0.4,0.3,0.2,0.1],[0.8,0.1,0.05,0.025,0.025],1000))
##print(find_most_context('<<-dfd-dfs-->>',3))

def get_all(word):
    index1=word.index("[")
    index2=word.index("]")
    result=''
    result=word[:index1]+word[index1+1:index2]+word[index2+1:]
    return result


def get_part(word):
    index1=word.index("[")
    index2=word.index("]")
    result=''
    result=word[:index1]+word[index2+1:]
    return result

print(get_part('zhang[大力开发经理说]绿色的房间里'))