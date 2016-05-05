# -*- coding: utf-8 -*-
import numpy as np
import vectorizer.utils as utils

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from vectorizer.english_stemmer import EnglishTokenizer
from sklearn.linear_model import LogisticRegression

tweets = None
classifications = None

with open('vectorizer/corpus/sanders/tweets.txt', 'r') as tweets_file:
    tweets = np.array(tweets_file.read().splitlines() )
    
with open('vectorizer/corpus/sanders/classification.txt', 'r') as classifications_file:
    classifications = np.array(classifications_file.read().splitlines())

def build_sentiment_no_sentiment():
    population = utils.prepare_entr_tweets(tweets, classifications, 2)

    classifier = LogisticRegression(C=2.5,
                                class_weight=None,
                                fit_intercept=True,
                                penalty='l2',
                                warm_start=True)

    pipeline = Pipeline([('tfidf', TfidfVectorizer(binary=True,
                                                   ngram_range=(1, 2),
                                                    smooth_idf=True,
                                                    stop_words='english',
                                                    sublinear_tf=True,
                                                    tokenizer=EnglishTokenizer(),
                                                    use_idf=True)),
                         ('clf', classifier)])
                         
    pipeline.fit(population['tweets'], y=population['classifications'])
    
    return pipeline

def build_positive_negative():
    population = utils.prepare_entr_tweets(tweets, classifications, 1)
    classifier = LogisticRegression(C=2,
                                    class_weight='balanced',
                                    fit_intercept=True,
                                    penalty='l2',
                                    warm_start=True)
    
    pipeline = Pipeline([('tfidf', TfidfVectorizer(binary=True,
                                                   ngram_range=(1, 1),
                                                    smooth_idf=False,
                                                    stop_words='english',
                                                    sublinear_tf=False,
                                                    tokenizer=EnglishTokenizer(),
                                                    use_idf=True)),
                         ('clf', classifier)])
                         
    pipeline.fit(population['tweets'], y=population['classifications'])
    
    return pipeline