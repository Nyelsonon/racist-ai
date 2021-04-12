import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
import numpy as np
import csv
import nltk, string
import csv
from csv import reader

query=input("tell me something: ")

print(query.lower())

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

print(query.lower().translate(remove_punctuation_map))
print(nltk.word_tokenize(query.lower().translate(remove_punctuation_map)))

stemmer = nltk.stem.porter.PorterStemmer()
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

print(stem_tokens(nltk.word_tokenize(query.lower().translate(remove_punctuation_map))))

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vctrz = TfidfVectorizer(ngram_range = (1,1),tokenizer = normalize, stop_words='english')

racism = []
filename = 'Tweets.csv'

with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        racism.append(repr(row))

vctrz.fit(racism)

query=query

tfidf_reports = vctrz.transform(racism).todense()
tfidf_question = vctrz.transform([query]).todense()

row_similarities = [1-spatial.distance.cosine(tfidf_reports[x],tfidf_question) for x in range(len(tfidf_reports))]

print(row_similarities)

print(racism[np.argmax(row_similarities)])

