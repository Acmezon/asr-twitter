import numpy as np
import nltk

from collections import Counter

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

