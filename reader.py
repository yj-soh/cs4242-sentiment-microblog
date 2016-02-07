import json
import csv

TWEET_DIR = 'data/tweets/'
TRAINING = 'data/training.csv'
DEVELOPMENT = 'data/development.csv'
TESTING = 'data/testing.csv'
TWEETS_TO_READ = TRAINING

def __read_json(filename):
    return json.load(open(filename))

def __read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for row in reader:
            yield row

def read():
    files = (row[2] + '.json' for row in __read_csv(TWEETS_TO_READ))
    jsons = (__read_json(TWEET_DIR + file) for file in files)
    return (json['text'] for json in jsons)
