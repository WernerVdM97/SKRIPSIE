import nltk
import torch.nn.functional as F
import numpy as np
import torch
from nltk.corpus import stopwords

#processing tokens
import unicodedata 
import re
import inflect 

#tokenize
from nltk import word_tokenize, sent_tokenize

#find ngrams
from nltk.util import ngrams

import time

############################word processing functions############################################
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
        if word == '.':
            new_words.append('</s>')
            new_words.append('<s>')
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def normalize(words,stop = False):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    if stop == True:
        words = remove_stopwords(words)
    return words
#######################################################################

#########################read file and process##############################################
def ReadAndProcess(path, tupleSize, stop):
    print('Loading file...')
    start = time.time()

    file = open(path).read()
    tokens = word_tokenize(file)

    end = time.time()
    print(end-start)

    print('Pre-processing...')
    start = time.time()

    data_tokens = normalize(tokens, stop)
    data_tokens.insert(0,'<s>')
    data_tokens.pop(-1)

    #create trigram and bigram frequency distribution from file
    myngrams = ngrams(data_tokens, tupleSize)
    ng_list = list(myngrams)

    #remove </s> followed by <s> cases
    x = 0
    while x < len(ng_list):
        for y in range(len(ng_list[x])-1):
            if ng_list[x][y] == '</s>' and ng_list[x][y+1] == '<s>':
                ng_list.pop(x)
                x = x - 1   
        x=x+1

    fd = nltk.FreqDist(ng_list)
    end = time.time()

    print(end-start)

    return fd
#######################################################################
