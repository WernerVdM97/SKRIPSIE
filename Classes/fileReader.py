import nltk
import torch.nn.functional as F
import numpy as np
import torch

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


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    #words = remove_stopwords(words)
    return words
#######################################################################

#########################read file and process##############################################
def ReadAndProcess(path):
    file = open(path).read()

    start = time.time()
    sentences = sent_tokenize(file)
    tokenised_sentences = []
    tokens = word_tokenize(file)

    # now loop over each sentence and tokenize it separately
    for sentence in sentences:
        tokenised_sentences.append(word_tokenize(sentence))

    end = time.time()
    print('Loading file...')
    print(end-start)

    start = time.time()
    data_sen = []

    for x in range(len(tokenised_sentences)):
        data_sen.append(normalize(tokenised_sentences[x]))

    data_tokens = normalize(tokens)

    #create frequency distribution from file
    trigrams = ngrams(data_tokens, 3)
    tg_list = list(trigrams)
    fd = nltk.FreqDist(tg_list)

    end = time.time()

    print('Pre-processing...')
    print(end-start)

    return fd
#######################################################################
