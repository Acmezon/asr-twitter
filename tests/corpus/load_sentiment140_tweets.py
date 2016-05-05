# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../preprocessing')

import csv
import io

from tweet import sanitize


with io.open('sentiment140/tweets.csv', 'rt', encoding='latin-1') as corpus:
    reader = csv.reader(corpus, delimiter=',', quotechar='"')
    with open('sentiment140/tweets.txt', 'w') as tweets:
        with open('sentiment140/classification.txt', 'w') as classification:
            for row in reader:
                tweet_text = row[-1]
                print(tweet_text.replace('\n', ' ').replace('\r', ''), file=tweets)
                
                sentiment = int(row[0])
                
                if sentiment == 0:
                    sentiment = 'negative'
                elif sentiment == 2:
                    sentiment = 'neutral'
                else:
                    sentiment = 'positive'

                print(sentiment, file=classification)

tweets = []
with open('sentiment140/tweets.txt', 'r') as file_tweets:
    for line in file_tweets:
        tweets.append(sanitize(line))

with open('sentiment140/tweets.txt', 'w') as file_tweets:
    for tweet in tweets:
        file_tweets.write(tweet + "\n")