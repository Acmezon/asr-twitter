import matplotlib.pyplot as plt
import numpy as np
import utils

from custom_vectorizer import Vectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def run(tweets, classifications):
    if len(tweets)!=len(classifications):
        raise ValueError('Error: Tweet population size and classifications size not matching.')
    
    population = utils.prepare_entr_tweets(tweets, classifications, 2)

    pipeline = Pipeline([('sent', Vectorizer(stop_words='english', ngram_range=(1, 3))),
                   ('clf', MultinomialNB())])
    parameters = {'clf__alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]}

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=0,
			cv=ShuffleSplit(population['tweets'].size))
   
    grid_search.fit(population['tweets'], population['classifications'])
    
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
    
if __name__ == "__main__":
    tweets = None
    classifications = None
    
    with open('../corpus/sanders/tweets.txt', 'r') as tweets_file:
        tweets = np.array(tweets_file.read().splitlines() )
        
    with open('../corpus/sanders/classification.txt', 'r') as classifications_file:
        classifications = np.array(classifications_file.read().splitlines())
    
    run(tweets, classifications)