import pandas as pd
import numpy as np
import nltk
import spacy
import regex as re

from config import *

def abbrev_conversion(text,lookup_dict):

  if text is None:
    print("Input text can be null")
  else:

    #Convert to lower case
    text = text.lower()
    words = text.split()
    abbrevs_removed = []

    for i in words:
        if i in lookup_dict:
            i = lookup_dict[i]
        abbrevs_removed.append(i)

  return ' '.join(abbrevs_removed)

  

  def cleanData_for_gensim(text):

    #pattern for identifying urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    #pattern for identifying html tags
    html_pattern = re.compile('<.*?>')
    #remove break line, tabs etc
    text = re.sub( '/(\r\n)+|\r+|\n+|\t+/', ' ', text)
    #remove special characters
    text= re.sub('[^\w\s]', ' ', text)
    #remove extra spaces
    text = re.sub(' +', ' ', text)
    #remove urls
    text= url_pattern.sub(r'', text)
    #remove html tags
    text= html_pattern.sub(r'', text)
    #Convert to spacy object
    doc = nlp(text)
    #Remove stopwords, punctuation, digit characters
    #return lemmatized tokens
    tokens = [token.lemma_ for token in doc if (token.is_stop == False and token.is_punct == False)]

    def remove_digits(lst):
      return [reduce(lambda x, y: x+y, filter(lambda x: not x.isdigit(), s), '') for s in lst]

    tokens = remove_digits(tokens)
    #return the tokens
    tokens = [x for x in tokens if x != '']
    return " ".join(tokens)


def cleanData_for_bert(text):
    #pattern for identifying urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    
    #pattern for identifying html tags
    html_pattern = re.compile('<.*?>')

    #remove break line, tabs etc
    text = re.sub( '/(\r\n)+|\r+|\n+|\t+/', ' ', text)
    #remove special characters
    text= re.sub('[^\w\s]', ' ', text)
    #remove extra spaces
    text = re.sub(' +', ' ', text)
    #remove urls
    text= url_pattern.sub(r'', text)
    #remove html tags
    text= html_pattern.sub(r'', text)
    # Match all digits in the string and replace them with an empty string
    text = ''.join((x for x in text if not x.isdigit()))

    return text








