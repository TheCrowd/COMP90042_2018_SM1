import nltk,operator
nltk.download("brown")
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma
import re
def isWord(word):
    result=re.match(r'[a-zA-Z]+',word)
    if result is None:
        return False
    else:
        return True
    
docs=[]
for para in brown.paras():
	docs.append(para)


doc_set_list=[]
doc_freq={}
for para in docs:
    doc_set=set()
    for sentence in para:
        for word in sentence:
            if isWord(word):
                word=word.lower()
                word=lemmatize(word)
                doc_set.add(word)
    doc_set_list.append(doc_set)
for item in doc_set_list:
    for word in item:
        doc_freq[word]=doc_freq.get(word,0)+1

words_file=open('combined.tab','r',encoding="utf8")
line_num=0;
word_similarity=[]
for line in words_file:
    if line_num==0:
        line_num+=1
        continue
    else:
        line_parts=line.split()
        word_similarity.append(line_parts)

def first_filter(words,doc_freq):
    new_words=[]
    for item in words:
        if(doc_freq.get(item[0],0)>=10 and doc_freq.get(item[1],0)>=10):
            new_words.append(item)
    return new_words
word_similarity=first_filter(word_similarity,doc_freq)
print(word in word_similarity)
from nltk.corpus import wordnet as wn
#this function returns the most common sense and second common sense of synsets
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
    

# return a boolean value, if the input word has a singe primary sense,then return True,else return False
def single_sense(word):
    noun_str='.n.'
    synsets=wn.synsets(word)
    if len(synsets)==1 and noun_str in synsets[0].name():
        return True,synsets[0].name()
    most_count,second_count=lemma_count(word,synsets)
    if most_count[1]>=5 and most_count[1]>=5*second_count[1] and noun_str in most_count[0]:
        return True,most_count[0]
    return False,None
    
def second_filter(words):
    word_sense=[]
    word_similarity=[]    
    for item in words:
        word_pair={}
        bool1,sense1=single_sense(item[0])
        bool2,sense2=single_sense(item[1])
        if bool1 and bool2:
            word_pair[item[0]]=sense1
            word_pair[item[1]]=sense2
            word_sense.append(word_pair)
            word_similarity.append(item)
    return word_sense,word_similarity

word_sense,word_similarity=second_filter(word_similarity)
for word in word_similarity:
    print(word)
#q2
word_wp_score={}
for word_pair in word_sense:
    word1=list(word_pair.keys())[0]
    word2=list(word_pair.keys())[1]
    word_wp_score[word1,word2]=wn.synset(word_pair[word1]).wup_similarity(wn.synset(word_pair[word2]))
print('------WP_similarity--------------')
word_wpscore_list=[(k,v) for k,v in word_wp_score.items()]
for score in word_wpscore_list:
    print(score)
    
#q3
from math import log
def cal_PPMI(word1,word2):
    sum_x=0
    sum_y=0
    sum_xy=0.0
    doc_len=0
    for para in doc_set_list:
        doc_len=doc_len+len(para)
        if word1 in para:
            sum_x+=1
        if word2 in para:
            sum_y+=1
        if word1 in para and word2 in para:
            sum_xy+=1
    if sum_xy==0:
        return 0
    return log(sum_xy*doc_len/(sum_x*sum_y),2)

word_PMI={}
for pair in word_similarity:
    word_PMI[pair[0],pair[1]]=cal_PPMI(word1=pair[0],word2=pair[1])
print('-----PPMI------------')
word_PMI_list=[(k,v) for k,v in word_PMI.items()]
for pmi in word_PMI_list:    
    print(pmi)

#q4
#import numpy as np
#test_word_set=set()
#for pair in word_similarity:
#    test_word_set.add(pair[0])
#    test_word_set.add(pair[1])
#test_word_list=list(test_word_set)
#brown_matrix=[]
#for word in test_word_list:
#    matrix_line=[]
#    for para in doc_set_list:
#        if word in para:
#            matrix_line.append(1)
#        else:
#            matrix_line.append(0)
#    brown_matrix.append(matrix_line)
#brown_matrix=np.matrix(brown_matrix)

from sklearn.feature_extraction import DictVectorizer
def get_BOW(word_set):
    BOW={}
    for word in word_set:
        BOW[word]=BOW.get(word,0)+1
    return BOW
doc_BOW=[]
for para in doc_set_list:
    doc_BOW.append(get_BOW(para))
vectorizer = DictVectorizer()
brown_matrix = vectorizer.fit_transform(doc_BOW).transpose()
word_list=vectorizer.get_feature_names()
################
#import numpy as np
#word_list=list(doc_freq.keys())
#brown_matrix=[]
#for word in word_list:
#    matrix_line=[]
#    for para in doc_set_list:
#        if word in para:
#            matrix_line.append(1)
#        else:
#            matrix_line.append(0)
#    brown_matrix.append(matrix_line)
#brown_matrix=np.matrix(brown_matrix)
        
    
###############
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=500)
brown_matrix = svd.fit_transform(brown_matrix)

from scipy.spatial.distance import cosine as cos_distance
def cal_cosine(word1,word2):
    index1=word_list.index(word1)
    index2=word_list.index(word2)
    vect1=brown_matrix[index1]
    vect2=brown_matrix[index2]
    return 1-cos_distance(vect1,vect2)
word_cos={}
for pair in word_similarity:
    word_cos[pair[0],pair[1]]=cal_cosine(pair[0],pair[1])
print('------cosine similarity---------')
word_cos_list=[(k,v) for k,v in word_cos.items()]
for cos in word_cos_list:    
    print(cos)
#q5
from gensim.models import Word2Vec
sents=brown.sents()
model = Word2Vec(sents, size=500, window=6, min_count=5, workers=4,iter=50)
word_gensim={}
for pair in word_similarity:
    word_gensim[pair[0],pair[1]]=model.wv.similarity(pair[0],pair[1])
print('------gensim_similarity---------')
word_gensim_list=[(k,v) for k,v in word_gensim.items()]
for line in word_gensim_list:    
    print(line)


#q6
from scipy import stats
def norm_list(aList):
    new_list=[]
    for line in aList:
        list_parts=[]
        list_temp=list(line[0])
        list_temp.sort()
        list_parts.append(list_temp)
        list_parts.append(line[1])
        new_list.append(list_parts)
    new_list=sorted(new_list, key=lambda x:x[0][0])
    final_list=[]
    for line in new_list:
        final_list.append(line[1])
    return final_list
test_list=[]
for word in word_similarity:
    line_part=[word[0],word[1]]
    line_parts=[line_part,float(word[2])]
    test_list.append(line_parts)
test_list=norm_list(test_list)
word_gensim_list=norm_list(word_gensim_list)
word_cos_list=norm_list(word_cos_list)
word_wpscore_list=norm_list(word_wpscore_list)
word_PMI_list=norm_list(word_PMI_list)
print()
print('-------golden standard vs Wu_Palmer Similarity-------')
print(stats.pearsonr(test_list,word_wpscore_list)[0])
print()
print('-------golden standard vs PPMI-------')
print(stats.pearsonr(test_list,word_PMI_list)[0])
print()
print('-------golden standard vs cosine similarity-------')
print(stats.pearsonr(test_list,word_cos_list)[0])
print()
print('-------golden standard vs gensim-------')
print(stats.pearsonr(test_list,word_gensim_list)[0])
