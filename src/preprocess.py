import os
import re
import sys
import string
from nltk.stem import PorterStemmer

def replace(tokens, to_replace, replace_by):
    '''
    --replace a particular token by another

    --Parameters:
        tokens: list of tokens
        to_replace: token to replace
        replace_by: replacement token
    
    --Returns:
        list of tokens
    '''
    for i,token in enumerate(tokens):
        if token == to_replace:
            tokens[i] = replace_by
    return tokens


def replace_num(sentence, replace_by = "NUM"):
    '''
    --replace numbers like years with NUM tag

    --Parameters:
        sentence: string
        replace_by: replacement token (NUM)

    --Returns:
        list of tokens
    '''
    tokens = sentence.split()
    for i,token in enumerate(tokens):
        if token.isnumeric():
            tokens[i] = replace_by
    return " ".join(tokens)


def replace_apostrophe(sentence):
    '''
    --replace it's with it is and 's with space

    --Parameters:
        sentence: string
    
    --Returns:
        sentence: string
    '''
    return sentence.replace("it's", "it").replace("'s", "")

def replace_nt(sentence):
    '''
    --replace tokens according to the mapping

    {
        "cant":"can not",
        "couldnt": "could not",
        "wouldnt": "would not",
        "wont": "will not",
        "didnt": "did not",
        "dont": "do not"
        "shouldnt": "should not",
        "shant": "shall not"
        "aint": "am not",
        "arent": "are not",
        "havent": "have not",
        "hadnt": "had not",
        "isnt": "is not",
    }

    --Parameters:
        sentence: string

    --Returns:
        sentence: string

    '''
    mapping = {
        "cant":"can not",
        "couldnt": "could not",
        "wouldnt": "would not",
        "wont": "will not",
        "didnt": "did not",
        "dont": "do not",
        "shouldnt": "should not",
        "shant": "shall not",
        "aint": "am not",
        "arent": "are not",
        "havent": "have not",
        "hadnt": "had not",
        "isnt": "is not",
    }

    for key, value in mapping.items():
        sentence = sentence.replace(key, value)

    return sentence


def normalize_sentence(sentence):
    '''
    --perform stemming using PorterStemmer

    --Parameters:
        sentence: string
    
    --Returns:
        sentence
    '''
    tokens = sentence.split()
    stemmer = PorterStemmer()
    return " ".join(list(map(lambda x: stemmer.stem(x), tokens)))


def tokenize(query, num="NUM"):
    '''
    --preprocess and tokenize

    --Parameters:
        query: string

    --Returns:
        tokens: list of strings

    '''
    sentence = query.lower()
    sentence = replace_apostrophe(sentence)
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = replace_nt(sentence)
    sentence = replace_num(sentence, num)
    sentence = normalize_sentence(sentence)
    return sentence.split()


if __name__ == "__main__":
    sentence = "He's been working on an amazing movie. Isn't he?"
    print(tokenize(sentence))