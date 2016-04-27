import math
import numpy as np
import nltk

from collections import Counter
from nltk.corpus import sentiwordnet as swn
from sklearn.preprocessing import label_binarize
from random import shuffle

def prepare_entr_tweets(tweets, classifications, subtask, test_size=0.7):

	if len(tweets)!=len(classifications):
		raise ValueError('Error: Tweet population size and classifications size not matching.')

	index_shuf = list(range(len(tweets)))
	shuffle(index_shuf)

	tweets_shuf = [tweets[i] for i in index_shuf]
	class_shuf = [classifications[i] for i in index_shuf]
	
	tweets = np.array(tweets_shuf)
	classifications = np.array(class_shuf)

	if subtask == 1:
		neutral_indices = np.where(classifications == 'neutral')[0]
		tweets = np.delete(tweets, neutral_indices)
		classifications = np.delete(classifications, neutral_indices)
		
		frontier_index = math.floor(tweets.size * test_size)
		
		train_tweets = tweets[:frontier_index]
		train_classif = classifications[:frontier_index]
		
		val_tweets = tweets[frontier_index:]
		val_classif = classifications[frontier_index:]
		
		val_classif_bin = label_binarize(val_classif, ['positive', 'negative'])

	elif subtask == 2:
		classifications[np.where(classifications == 'positive')[0]] = 's'
		classifications[np.where(classifications == 'negative')[0]] = 's'
		classifications[np.where(classifications == 'neutral')[0]] = 'ns'
		
		frontier_index = math.floor(tweets.size * test_size)        
		
		train_tweets = tweets[:frontier_index]
		train_classif = classifications[:frontier_index]
		
		val_tweets = tweets[frontier_index:]
		val_classif = classifications[frontier_index:]
		
		val_classif_bin = label_binarize(val_classif, ['s', 'ns'])

	elif subtask == 3:
		classifications[np.where(classifications == 'negative')[0]] = 'np'
		classifications[np.where(classifications == 'neutral')[0]] = 'np'
		
		frontier_index = math.floor(tweets.size * test_size)
		
		train_tweets = tweets[:frontier_index]
		train_classif = classifications[:frontier_index]
		
		val_tweets = tweets[frontier_index:]
		val_classif = classifications[frontier_index:]
		
		val_classif_bin = label_binarize(val_classif, ['positive', 'np'])

	elif subtask == 4:
		classifications[np.where(classifications == 'positive')[0]] = 'nn'
		classifications[np.where(classifications == 'neutral')[0]] = 'nn'
		
		frontier_index = math.floor(tweets.size * test_size)
		
		train_tweets = tweets[:frontier_index]
		train_classif = classifications[:frontier_index]
		
		val_tweets = tweets[frontier_index:]
		val_classif = classifications[frontier_index:]
		
		val_classif_bin = label_binarize(val_classif, ['nn', 'negative'])

	else:
		raise ValueError('Error: Param subtask in [1,4].')

	return {'tweets' : tweets, 
			'classifications' : classifications,  
			'train_tweets' : train_tweets, 
			'train_classif' : train_classif, 
			'val_tweets' : val_tweets, 
			'val_classif' : val_classif, 
			'val_classif_bin' : val_classif_bin }

def get_tweets():
	with open('data/tweets.txt', 'r') as tweets_file:
		tweets = np.array(tweets_file.read().splitlines() )
		
	with open('data/classification.txt', 'r') as classifications_file:
		classifications = np.array(classifications_file.read().splitlines())

	if len(tweets)!=len(classifications):
		raise ValueError('Error: Tweet population size and classifications size not matching.')

	return (tweets, classifications)

def get_sentiment(word):
	synsets = list(swn.senti_synsets(word.lower()))
	pos_score = sum(ss.pos_score() for ss in synsets)/float(len(synsets))
	neg_score = sum(ss.neg_score() for ss in synsets)/float(len(synsets))
	obj_score = sum(ss.obj_score() for ss in synsets)/float(len(synsets))
	return {'pos_score': pos_score,
			'neg_score': neg_score,
			'obj_score': obj_score}

part_of_speech = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
def count_pos(text):
	tokens = nltk.word_tokenize(text.lower())
	text = nltk.Text(tokens)
	tagged = nltk.pos_tag(text, tagset='universal')
	counts = Counter(tag for word,tag in tagged)
	total = sum(counts.values())
	res = dict((word, float(count)/total) for word,count in counts.items())
	for pos in part_of_speech:
		if pos not in res:
			res[pos] = 0
	return res