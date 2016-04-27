import matplotlib.pyplot as plt
import utils

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit
from english_stemmer import EnglishTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def run(tweets, classifications, subtask, grid_search=True):

	if len(tweets)!=len(classifications):
		raise ValueError('Error: Tweet population size and classifications size not matching.')
	
	population = utils.prepare_entr_tweets(tweets, classifications, subtask)

	pipeline = Pipeline([('tfidf', TfidfVectorizer()),
				   ('clf', MultinomialNB())])

	if grid_search:
		
		parameters = {'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
						'tfidf__stop_words': ['english'],
						'tfidf__smooth_idf': [True, False],
						'tfidf__use_idf': [True, False],
						'tfidf__sublinear_tf': [True, False],
						'tfidf__binary': [True, False],
						'tfidf__tokenizer': [EnglishTokenizer()],
						'clf__alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
					 }

		grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=0,
			cv=ShuffleSplit(population['tweets'].size))
		
		grid_search.fit(population['tweets'], population['classifications'])

		print("Best score: %0.3f" % grid_search.best_score_)
		print("Best parameters set:")
		best_parameters = grid_search.best_estimator_.get_params()
		for param_name in sorted(parameters.keys()):
			print("\t%s: %r" % (param_name, best_parameters[param_name]))

		pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words=best_parameters['tfidf__stop_words'],
			tokenizer=best_parameters['tfidf__tokenizer'], ngram_range=best_parameters['tfidf__ngram_range'],
			smooth_idf=best_parameters['tfidf__smooth_idf'], use_idf=best_parameters['tfidf__use_idf'],
			sublinear_tf=best_parameters['tfidf__sublinear_tf'], binary=best_parameters['tfidf__binary'])),
				   ('clf', MultinomialNB(alpha=best_parameters['clf__alpha']))])

	else:

		tfidf = None   
		multinomibalnb = None
		if subtask == 1:
			tfidf = TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 2), sublinear_tf=True)
			multinomibalnb = MultinomialNB(alpha=0.5)
		elif subtask == 2:
			tfidf = TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 3), use_idf=False)
			multinomibalnb = MultinomialNB(alpha=0.1)
		elif subtask == 3:
			tfidf = TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 2), smooth_idf=False, use_idf=False)
			multinomibalnb = MultinomialNB(alpha=0.05)
		elif subtask == 4:
			tfidf = TfidfVectorizer(stop_words='english', tokenizer=EnglishTokenizer(), ngram_range=(1, 3), binary=True, smooth_idf=False, use_idf=False)
			multinomibalnb = MultinomialNB(alpha=0.05)
		else:
			raise ValueError('Error: Param subtask in [1,4].')

		pipeline = Pipeline([('tfidf', tfidf), ('multinomialnaive', multinomibalnb)])

	pipeline.fit(population['train_tweets'], y=population['train_classif'])
	
	predicted = pipeline.predict(population['val_tweets'])
	
	metrics = precision_recall_fscore_support(population['val_classif'], predicted, average='macro', pos_label=None)
	
	print("Exactitud:{0}\nPrecision:{1}\nRecall:{2}\nF1:{3}".format(
		accuracy_score(population['val_classif'], predicted), metrics[0], metrics[1], metrics[2]))
	
	score = pipeline.predict_proba(population['val_tweets'])[:, 0]

	print("AUC:{0}".format(average_precision_score(population['val_classif_bin'], score, average="micro")))
	
	precision = dict()
	recall = dict()
	average_precision = dict()
	
	# Compute micro-average ROC curve and ROC area
	precision["micro"], recall["micro"], _ = precision_recall_curve(
												population['val_classif_bin'], score)
	average_precision["micro"] = average_precision_score(population['val_classif_bin'], score,
														 average="micro")
	
	# Plot Precision-Recall curve for each class
	plt.clf()
	plt.plot(recall["micro"], precision["micro"],
			 label='Precision-recall curve (area = {0:0.2f})'
				   ''.format(average_precision["micro"]))
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve')
	plt.legend(loc="lower right")
	plt.show()