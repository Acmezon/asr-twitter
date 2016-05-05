# -*- coding: utf-8 -*-
import csv
import json
import os

from tweet import sanitize


with open('sanders/corpus.csv', 'r') as corpus:
    reader = csv.reader(corpus, delimiter=',', quotechar='"')
    
    with open('sanders/tweets.txt', 'w') as tweets:
        with open('sanders/classification.txt', 'w') as classification:
            for row in reader:
                if row[1] == 'irrelevant':
                    continue
                filename = row[-1]
                if os.path.isfile('sanders/rawdata/{0}.json'.format(filename)):
                    with open('sanders/rawdata/{0}.json'.format(filename), 'r') as tweet_file:
                        tweet = json.load(tweet_file)
                        print(tweet['text'].replace('\n', ' ').replace('\r', ''), file=tweets)
                        print(row[1], file=classification)
                    
tweets = []
with open('sanders/tweets.txt', 'r') as file_tweets:
    for line in file_tweets:
        tweets.append(sanitize(line))


with open('sanders/tweets.txt', 'w') as file_tweets:
    for i, tweet in enumerate(tweets):
        file_tweets.write(tweet + "\n")