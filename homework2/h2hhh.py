import nltk
import math
import sklearn 
from sklearn.feature_extraction import DictVectorizer
#nltk.download('brown')
from nltk.corpus import brown
from nltk.corpus import wordnet as wn

#lemmatise
#nltk.download('wordnet')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def get_BOW(text):
    BOW = {}
    for wordset in text:
        for word in wordset:
            BOW[word] = BOW.get(word,0) + 1
    return BOW

docs = brown.paras()

wanted_list = []
for paragraphs in docs:
    s = set()
    for sentences in paragraphs:
        for word in sentences:
            l = lemmatize(word).lower()
            s.add(l)
    wanted_list.append(s)
    
word_freq_mapping = get_BOW(wanted_list)

f = open('./combined.tab','r')
line_num = 0
conbined_tab = []
for line in f:
    line_num += 1
    if line_num == 1:
        continue
    else:
        line_parts=line.split()
        conbined_tab.append(line_parts)
#f.close()
##    data = fin.read().splitlines(True)
#f = open('./combined.tab','w').writelines(lines[1:])

def freq_over_ten(word):
    for elem in word_freq_mapping:
        if word == elem and word_freq_mapping[elem] >= 10:
            return True

    

list_v1 = []
for line in conbined_tab:
#    wds = line.split()
    if freq_over_ten(line[0]) is True and freq_over_ten(line[1]) is True:
        list_v1.append(line)

#count =0
#for line in list_v1:
#    count += 1
#    print line
#print count

#print fil_one()
#####end of filter one

def isValid(word):
    dictionary={}
    synsets = wn.synsets(word)
    if (len(synsets) == 1) and ('.n.' in  synsets[0].name()):
        dictionary[word] = synsets[0].name()
        return [True, dictionary]
    else: 
        list_1 = []
        dict_1={}
        for synset in synsets:
            dict_1[synset] = 0
            for lemma in synset.lemmas():
                if word == lemma.name():
                    dict_1[synset] = dict_1.get(synset,0)+lemma.count() 
                else:
                    None
#                print synset, dict[synset]
        for key, value in sorted(dict_1.iteritems(), key = lambda(k,v):(v,k), reverse = True):
            list_1.append([key, value])
        if ('.n.' in list_1[0][0].name()) and list_1[0][1] >= 5 and list_1[0][1] >= 5 * list_1[1][1]:
            dictionary[word] = list_1[0][0].name()
            return [True, dictionary]
    return [False, None]
#print isValid('police')
#print isValid('man')
#    

list_v2 = []
word_synset={}
for line in list_v1:
    bool1, dic1 = isValid(line[0])
    bool2, dic2 = isValid(line[1])
    if bool1 and bool2:
        list_v2.append(line)
        word_synset[line[0]]=dic1[line[0]] 
        word_synset[line[1]]=dic2[line[1]]

print ('filtered test set')
for line in list_v2:
    print line 

#####q2

wu_pal_dict={}
print ('Wu Palmer Similarity')
for line in list_v2:
    similarity = wn.synset(str(word_synset[line[0]])).wup_similarity(wn.synset(str(word_synset[line[1]])))
    wu_pal_dict[line[0], line[1]]= similarity  
    print line[0], line[1], similarity

#####q3
    
def get_PPMI(word1, word2):
    count1 = 0
    count2 = 0
    count_both = 0
    total_count = 0.0
    for elem in wanted_list:
        total_count += len(elem)
        if word1 in elem:
            count1 += 1
        if word2 in elem:
            count2 += 1
        if (word1 in elem) and (word2 in elem):
            count_both += 1
    if count_both == 0:
        return 0
    return math.log((count_both/total_count)/((count1/total_count)*(count2/total_count)), 2)

print ('PPMI dictionary')
ppmi_dict = {}
for line in list_v2:
    ppmi_dict[line[0], line[1]] = get_PPMI(line[0], line[1])
    print line[0], line[1], get_PPMI(line[0], line[1])

#####q4
def get_BOW_v1(text):
    BOW = {}
    for word in text:
        BOW[word] = BOW.get(word,0) + 1
    return BOW

list_vect = []
for elem in wanted_list:
    list_vect.append(get_BOW_v1(elem))
    
vectorizer = DictVectorizer()
dataset = vectorizer.fit_transform(list_vect).transpose()
feature_name = vectorizer.get_feature_names()

from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine as cos_distance

svd = TruncatedSVD(n_components=500)
brown_matrix = svd.fit_transform(dataset)

print ('Cosine similarity dictionary')
cos_dict={}
for elem in list_v2:
    words1_index = feature_name.index(elem[0])
    words2_index = feature_name.index(elem[1])
    cos_similarity = 1 - cos_distance(brown_matrix[words1_index], brown_matrix[words2_index])
    cos_dict[elem[0], elem[1]] = cos_similarity
    print elem[0], elem[1], cos_similarity

#####q5
from gensim.models import Word2Vec
sent = brown.sents()
model = Word2Vec(sent, size = 500, window = 6, workers = 4, min_count = 5,iter = 50)
print ('Gensim dictionary')
gensim_dict = {}
for elem in list_v2:
    gensim_dict [elem[0], elem[1]] = model.wv.similarity(elem[0], elem[1])
    print elem[0], elem[1], model.wv.similarity(elem[0], elem[1])

######q6
from scipy import stats
#def dict_to_list (dictionary):
#    list_dict = []
#    for key, value in dictionary:
#        list_dict.append([key,value])
#    return list_dict

list_third_rows = []
for elem in list_v2:
    list_third_rows.append(float(elem[2]))
    
list_wu = wu_pal_dict.values()
list_ppmi = ppmi_dict.values()
list_cos = cos_dict.values()
list_gen = gensim_dict.values()
print '-------gold standard vs Wu Palmer similarity-----'
print stats.pearsonr(list_third_rows, list_wu)[0]
print '-------gold standard vs PPMI --------------------'
print stats.pearsonr(list_third_rows, list_ppmi)[0]
print '-------gold standard vs cosine similarity--------'
print stats.pearsonr(list_third_rows, list_cos)[0]
print '-------gold standard vs gensim similarity--------'
print stats.pearsonr(list_third_rows, list_gen)[0]

