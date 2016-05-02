import utils
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from english_stemmer import EnglishTokenizer
from sklearn.metrics import precision_recall_fscore_support

def test(tweets, classifications, classifier, classifier_parameters):
    if len(tweets)!=len(classifications):
        raise ValueError('Error: Tweet population size and classifications size not matching.')
    
    for i in range(1, 3):
        with open('test_results.txt', 'a') as results_file:
            if i == 1:
                print("Positivo-Negativo. Clasificador: {0}".format(type(classifier).__name__), file=results_file)
                print("Positivo-Negativo. Clasificador: {0}".format(type(classifier).__name__))
            else:
                print("Sentimiento-sin sentimiento. Clasificador: {0}".format(type(classifier).__name__), file=results_file)
                print("Sentimiento-sin sentimiento. Clasificador: {0}".format(type(classifier).__name__))
        
        population = utils.prepare_entr_tweets(tweets, classifications, i)
    
        pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                       ('clf', classifier)])
    
        parameters = {'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                        'tfidf__stop_words': ['english'],
                        'tfidf__smooth_idf': [True, False],
                        'tfidf__use_idf': [True, False],
                        'tfidf__sublinear_tf': [True, False],
                        'tfidf__binary': [True, False],
                        'tfidf__tokenizer': [EnglishTokenizer()],
                     }
    
        for key, value in classifier_parameters.items():
            parameters['clf__{0}'.format(key)] = value
        
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1,
            cv=KFold(population['tweets'].size, n_folds=6, shuffle=True))
        
        grid_search.fit(population['tweets'], population['classifications'])
        
        best_parameters = grid_search.best_estimator_.get_params()
        
        with open('test_results.txt', 'a') as results_file:
            print("Best score: %0.3f" % grid_search.best_score_, file=results_file)
            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:", file=results_file)
            print("Best parameters set:")
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]), file=results_file)
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
        if isinstance(classifier, MultinomialNB):
            clf = MultinomialNB(alpha=best_parameters['clf__alpha'])
        elif isinstance(classifier, SVC):
            clf = SVC(C=best_parameters['clf__C'],
                      kernel = best_parameters['clf__kernel'],
                      shrinking = best_parameters['clf__shrinking'],
                      decision_function_shape = best_parameters['clf__decision_function_shape'])
        elif isinstance(classifier, LogisticRegression):
            clf = LogisticRegression(penalty = best_parameters['clf__penalty'],
                                     C = best_parameters['clf__C'],
                                     fit_intercept = best_parameters['clf__fit_intercept'],
                                     class_weight = best_parameters['clf__class_weight'],
                                     warm_start = best_parameters['clf__warm_start'],
                                     solver = best_parameters['clf__solver'])
            
        pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words=best_parameters['tfidf__stop_words'],
    			tokenizer=best_parameters['tfidf__tokenizer'], ngram_range=best_parameters['tfidf__ngram_range'],
    			smooth_idf=best_parameters['tfidf__smooth_idf'], use_idf=best_parameters['tfidf__use_idf'],
    			sublinear_tf=best_parameters['tfidf__sublinear_tf'], binary=best_parameters['tfidf__binary'])),
    				   ('clf', clf)])
    
        pipeline.fit(population['train_tweets'], y=population['train_classif'])
        
        predicted = pipeline.predict(population['val_tweets'])
        
        metrics = precision_recall_fscore_support(population['val_classif'], predicted, average='macro', pos_label=None)
        
        with open('test_results.txt', 'a') as results_file:
            print("\nPrecision:{0}\nRecall:{1}\nF1:{2}\n".format(
                metrics[0], metrics[1], metrics[2]), file=results_file)

            print("\nPrecision:{0}\nRecall:{1}\nF1:{2}\n".format(
                metrics[0], metrics[1], metrics[2]))
    