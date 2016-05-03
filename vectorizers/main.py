# -*- coding: utf-8 -*-
import numpy as np

from test_tf_idf_vectorizer import test as test_tf_idf
from test_nltk_vectorizer import test as test_nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

tweets = None
classifications = None

with open('../corpus/sanders/tweets.txt', 'r') as tweets_file:
    tweets = np.array(tweets_file.read().splitlines() )
    
with open('../corpus/sanders/classification.txt', 'r') as classifications_file:
    classifications = np.array(classifications_file.read().splitlines())


###### TF-IDF ######

# with open('test_results.txt', 'a') as results_file:
#        print("##########################################\nTF-IDF Vectorizer tests\n##########################################\n\n", file=results_file)
#        print("##########################################\nTF-IDF Vectorizer tests\n##########################################\n\n")

multinomial_parameters = {
    'alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
}

svm_parameters = {
    'C': [0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'shrinking': [True, False],
    'decision_function_shape': ['ovo', 'ovr', None]
}

logreg_parameters = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5],
    'fit_intercept' : [True, False],
    'class_weight' : ['balanced', None],
    'warm_start' : [True, False]
}

#Multinomial Classifier
# test_tf_idf(tweets, classifications, MultinomialNB(), multinomial_parameters)

#SVM Classifier
# test_tf_idf(tweets, classifications, SVC(), svm_parameters)

#Logistic regression Classifier
# test_tf_idf(tweets, classifications, LogisticRegression(), logreg_parameters)


###### NLTK #######
# with open('test_results.txt', 'a') as results_file:
#    print("##########################################\nNLTK Vectorizer tests\n##########################################\n\n", file=results_file)
#    print("##########################################\nNLTK Vectorizer tests\n##########################################\n\n")

#Multinomial Classifier
# test_nltk(tweets, classifications, MultinomialNB(), multinomial_parameters)

#SVM Classifier
# test_nltk(tweets, classifications, SVC(), svm_parameters)

#Logistic regression Classifier
# test_nltk(tweets, classifications, LogisticRegression(), logreg_parameters)

###### Union #######
with open('test_results.txt', 'a') as results_file:
    print("##########################################\nUnion Vectorizer tests\n##########################################\n\n", file=results_file)
    print("##########################################\nUnion  Vectorizer tests\n##########################################\n\n")
    
#Multinomial Classifier
test_nltk(tweets, classifications, MultinomialNB(), multinomial_parameters)

#SVM Classifier
test_nltk(tweets, classifications, SVC(), svm_parameters)

#Logistic regression Classifier
test_nltk(tweets, classifications, LogisticRegression(), logreg_parameters)