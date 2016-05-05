# -*- coding: utf-8 -*-

import numpy as np

def prepare_entr_tweets(tweets, classifications, subtask):

	if len(tweets)!=len(classifications):
		raise ValueError('Error: Tweet population size and classifications size not matching.')

	tweets = np.array(tweets)
	classifications = np.array(classifications)

	if subtask == 1:
		neutral_indices = np.where(classifications == 'neutral')[0]
		tweets = np.delete(tweets, neutral_indices)
		classifications = np.delete(classifications, neutral_indices)

	elif subtask == 2:
		classifications[np.where(classifications == 'positive')[0]] = 's'
		classifications[np.where(classifications == 'negative')[0]] = 's'
		classifications[np.where(classifications == 'neutral')[0]] = 'ns'
	else:
		raise ValueError('Error: Param subtask in [1,2].')

	return {'tweets' : tweets, 
		  'classifications' : classifications
             }