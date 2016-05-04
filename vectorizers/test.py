import numpy as np
import utils

from NLTKVectorizer import NLTKVectorizer
from sklearn.pipeline import Pipeline
from english_stemmer import EnglishTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


tweets = None
classifications = None

with open('../corpus/sanders/tweets.txt', 'r') as tweets_file:
    tweets = np.array(tweets_file.read().splitlines() )
    
with open('../corpus/sanders/classification.txt', 'r') as classifications_file:
    classifications = np.array(classifications_file.read().splitlines())
    

population = utils.prepare_entr_tweets(tweets, classifications, 1)
classifier = LogisticRegression(C=2,
                                class_weight='balanced',
                                fit_intercept=True,
                                penalty='l1',
                                warm_start=True)

pipeline = Pipeline([('nltk', NLTKVectorizer(stop_words='english',
                                             binary=True,
                                             ngram_range=(1, 1))),
                     ('clf', classifier)])
                     
pipeline.fit(population['train_tweets'], y=population['train_classif'])
        
predicted = pipeline.predict(population['val_tweets'])

metrics = precision_recall_fscore_support(population['val_classif'], predicted, average='macro', pos_label=None)

with open('test_results.txt', 'a') as results_file:
    print("\nPrecision:{0}\nRecall:{1}\nF1:{2}\n".format(
        metrics[0], metrics[1], metrics[2]), file=results_file)    
    print("\nPrecision:{0}\nRecall:{1}\nF1:{2}\n".format(
        metrics[0], metrics[1], metrics[2]))

print("###############################################################")

population = utils.prepare_entr_tweets(tweets, classifications, 2)
classifier = LogisticRegression(C=2.5,
                                class_weight=None,
                                fit_intercept=True,
                                penalty='l2',
                                warm_start=True)

pipeline = Pipeline([('nltk', NLTKVectorizer(stop_words='english',
                                             binary=True,
                                             ngram_range=(1, 1))),
                     ('clf', classifier)])

pipeline.fit(population['train_tweets'], y=population['train_classif'])
        
predicted = pipeline.predict(population['val_tweets'])

metrics = precision_recall_fscore_support(population['val_classif'], predicted, average='macro', pos_label=None)

with open('test_results.txt', 'a') as results_file:
    print("\nPrecision:{0}\nRecall:{1}\nF1:{2}\n".format(
        metrics[0], metrics[1], metrics[2]), file=results_file)    
    print("\nPrecision:{0}\nRecall:{1}\nF1:{2}\n".format(
        metrics[0], metrics[1], metrics[2]))