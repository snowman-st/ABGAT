import os
import numpy as np
import random
from xml.etree.ElementTree import parse
import re
import json
import pickle


def check(x):
    return len(x) >= 1 and not x.isspace()

def tokenizer(text):
    tokens = [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
    return list(filter(check, tokens))

def parse_sentence(pathin,pathout):
    tree = parse(pathin)
    sentences = tree.getroot()
    sentence_count = 0
    f = open(pathout,'a+',encoding='utf-8')
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        sentence_count += 1
        f.write(text+'\n')
    f.close()
    print('{} sentences were saved in {}'.format(sentence_count,pathout))

def parse_sentence_term(path, lowercase=False):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    sentences_count = 0
    each_sentence_len = []
    d = {
        'neutral': 0,
        'positive': 1,
        'negative': 2
    }
    for sentence in sentences:
        sentences_count += 1
        each = 0
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        for aspectTerm in aspectTerms:
            term = aspectTerm.get('term')
            if lowercase:
                term = term.lower()
            polarity = aspectTerm.get('polarity')
            if polarity != 'conflict':
                each += 1
            start = aspectTerm.get('from')
            end = aspectTerm.get('to')
            piece = text + split_char + term + split_char + polarity + split_char + start + split_char + end
            data.append(piece)

    return data

if __name__=='__main__':
    parse_sentence("/home1/xk/document/datasets/SemEval-ABSA/ABSA-SemEval2014/Restaurants_Train.xml","datasets/semeval14/restaurant_sentences_train.raw")
    parse_sentence("/home1/xk/document/datasets/SemEval-ABSA/ABSA-SemEval2014/Restaurants_Test_Gold.xml","datasets/semeval14/restaurant_sentences_test.raw")
