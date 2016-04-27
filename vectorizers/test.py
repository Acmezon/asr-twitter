import vectorizer1
import utils

tweets, classifications = utils.get_tweets()

vectorizer1.run(tweets, classifications, subtask=1, grid_search=False)