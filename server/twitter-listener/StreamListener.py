# -*- coding: utf-8 -*-
import tweepy
from pymongo import MongoClient

#override tweepy.StreamListener to add logic to on_status
class TwitterListener(tweepy.StreamListener):
    def __init__(self, api):
        client = MongoClient()
        self.db = client['AS-Twitter']
        self.api = api

    def on_status(self, status):
        text = status.text
        date = status.created_at
        
        tweet = {
            'text': text,
            'date': date
        }        
        
        self.db.tweets.insert_one(tweet)
        
    def on_error(self, status_code):
        if status_code == 420:
            raise TimeoutError("Rate limit exceed")