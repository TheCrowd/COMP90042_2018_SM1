# allowing better python 2 & python 3 compatibility 
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

#hangman('whatever', human, 8, True)




#-------q1
from nltk.corpus import brown
import numpy
text = brown.words()
text = [c for c in text if c.isalpha()] # Assuming it really is a list of chars to a list of chars gives a list
brown_word_set = set()
for elem in text:
    brown_word_set.add (elem.lower())

#print (brown_word_list)

#print (brown_word_list)
brown_word_list=list(brown_word_set)
numpy.random.shuffle(brown_word_list)
test_set = brown_word_list[:1000]
print (len(test_set))
training_set = brown_word_list[1000:]
print (len(training_set))

#hangman(numpy.random.choice(test_set), human, 8, True)


#-------q2
def random_alpha_guess(mask, guessed, **kwargs):    
    alpha_list = ['a','b','c','d','e','f','g','h','i','j','k','l',
                  'm','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for word in guessed:
        if word in alpha_list:
            alpha_list.remove(word)
    single_alpha = numpy.random.choice(alpha_list)
    return single_alpha

#hangman(numpy.random.choice(test_set), random_alpha_guess, 26, False)
total_mistakes = 0
for elem in test_set:
    mistake = hangman(elem, random_alpha_guess, 26, False)
    total_mistakes += mistake

print (total_mistakes / len(test_set))

#--------q3
import operator

total_length = 0
alpha_dict = {}
for word in training_set:
    total_length += len(word)
    for alpha in word:
        alpha_dict[alpha] = (alpha_dict.get(alpha,0) + 1)
alpha_prob_dict = {}
for elem in alpha_dict.keys():
    alpha_prob_dict[elem] = alpha_dict[elem]/total_length
sorted_dict = sorted (alpha_prob_dict.items(), key = operator.itemgetter(1), reverse = True)


def unigram_guess(mask, guessed, **kwargs):
    prob_dict = kwargs['prob_dict']
    for elem in prob_dict:
        if elem[0] not in guessed:
            return elem[0]

total_unigram_mistakes = 0
for elem in test_set:
    mistake = hangman(elem, unigram_guess, 26, False, prob_dict = sorted_dict)
    total_unigram_mistakes += mistake

print (total_unigram_mistakes / len(test_set))

#--------------q4
adv_dict = {}
diff_len_dict = {}
for word in training_set:
    word_length=len(word)
    diff_len_dict[word_length] = diff_len_dict.get(word_length,0) + word_length
    alpha_dictionary = adv_dict.get(word_length, {})
    for alpha in word:
        alpha_dictionary[alpha] = alpha_dictionary.get(alpha,0) + 1
    adv_dict[word_length] = alpha_dictionary

alphabet_list = ['a','b','c','d','e','f','g','h','i','j','k','l',
                  'm','n','o','p','q','r','s','t','u','v','w','x','y','z']
for k, v in adv_dict.items():
    if len(adv_dict[k].keys()) != 26:
        for elem in alphabet_list:
            adv_dict[k][elem] = adv_dict[k].get(elem,0) + 1
            diff_len_dict[k] = diff_len_dict.get(k,0) + 1

for k in adv_dict.keys():
    for elem in adv_dict[k].keys():
        adv_dict[k][elem]=adv_dict[k][elem]/diff_len_dict[k]


for k in adv_dict.keys():
    adv_dict[k] = sorted (adv_dict[k].items(), key = operator.itemgetter(1), reverse = True)
adv_dict[0] = sorted_dict


def diff_len_unigram_guess(mask, guessed, **kwargs):
    prob_dict = kwargs['prob_dict']
    if len(mask) not in prob_dict.keys():
        for elem in prob_dict[0]:
            if elem[0] not in guessed:
                return elem[0]
    else:
        for elem in prob_dict[len(mask)]:
            if elem[0] not in guessed:
                return elem[0]

total_unigram_mistakes = 0
for elem in test_set:
    mistake = hangman(elem, diff_len_unigram_guess, 26, False, prob_dict = adv_dict)
    total_unigram_mistakes += mistake

print (total_unigram_mistakes / len(test_set))

#-----------------q5

import numpy as np
import math
def add_sentinel(word,n):
    beforestart=[]
    afterend=[]
    for i in range(n-1):
        beforestart.append('#')
        afterend.append('*')   
    return ''.join(beforestart)+word+''.join(afterend)

def get_ngram_counts(words,n):
    counts=[]
    for k in range(1,n+1):
        count={}
        for word in words:
            word=add_sentinel(word,5)
            if len(word)>=k:
                for i in range(len(word)-k+1):
                    keyStr="".join(word[i:i+k])
                    count[keyStr]=count.get(keyStr,0)+1
        counts.append(count)
    return counts



def most_left_context(word,n):
    blanks=[]
    for i in range(n):
        blanks.append('_')
    left_context=''.join(blanks)
    for i in range(0,len(word)-n):
        if word[i:i+n].count('_')<left_context.count('_') and word[i+n-1] is '_':
            left_context=word[i:i+n];
    return left_context

def get_log_prob(word,counts,lambdas,uni_total):
    gram_num=len(counts)
    probs=[]
    for i in range(gram_num):
        if i<gram_num-1:
            if counts[gram_num-1-i].get(word[i:],0)!=0:
                prob=counts[gram_num-1-i].get(word[i:],0)/counts[gram_num-2-i].get(word[i:-1],0)*lambdas[gram_num-1-i]
                probs.append(prob)
        elif i==gram_num-1:
            prob=counts[0].get(word[-1],0)/uni_total*lambdas[0]
            probs.append(prob)
    return math.log(sum(probs))
	
def get_all_log_prob(word,char,counts,lambdas, uni_total):
    blank_num=word.count('_')
    cur_pos=0       
    all_prob=[]
    for i in range(blank_num):
        pos_end=word[cur_pos:].find('_')
        if pos_end!=-1:
            pos_start=pos_end
            while word[pos_start-1]!='_':
                pos_start=pos_start-1
            log_prob=get_log_prob(word[pos_start:pos_end]+char,counts[0:pos_end-pos_start+1],lambdas,uni_total)
            all_prob.append(log_prob)
        cur_pos=pos_end+1
    return sum(all_prob)
    


def ngram_guess(mask,guessed,**kwargs):
    n=kwargs['n']
    counts=kwargs['counts']
    uni_total=kwargs['uni_total']
    lambdas=kwargs['lambdas']
    alphas = ['a','b','c','d','e','f','g','h','i','j','k','l',
                  'm','n','o','p','q','r','s','t','u','v','w','x','y','z']
    maskStr=add_sentinel(''.join(mask),n)
    gram_item=most_left_context(maskStr,n)
    for word in guessed:
        if word in alphas:
            alphas.remove(word)
    prob=float('-inf')
    guess=np.random.choice(alphas)      
    for alpha in alphas:
        cur_prob=get_all_log_prob(gram_item,alpha,counts,lambdas,uni_total)
        if cur_prob>prob:
            prob=cur_prob
            guess=alpha
    return guess


counts=get_ngram_counts(training_set,5)
uni_total=sum(counts[0].values())
lambdas=[0.01,0.02,0.07,0.2,0.7]    

    
total_unigram_mistakes = 0
for elem in test_set:
    mistake = hangman(elem, ngram_guess, 26, False, n=3,counts=counts,lambdas=lambdas,uni_total=uni_total)
    total_unigram_mistakes += mistake

print (total_unigram_mistakes / len(test_set))

total_unigram_mistakes = 0
for elem in test_set:
    mistake = hangman(elem, ngram_guess, 26, False, n=4,counts=counts,lambdas=lambdas,uni_total=uni_total)
    total_unigram_mistakes += mistake

print (total_unigram_mistakes / len(test_set))

total_unigram_mistakes = 0
for elem in test_set:
    mistake = hangman(elem, ngram_guess, 26, False, n=5,counts=counts,lambdas=lambdas,uni_total=uni_total)
    total_unigram_mistakes += mistake

print (total_unigram_mistakes / len(test_set))


























































