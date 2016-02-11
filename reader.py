import json
import csv

TWEET_DIR = 'data/tweets/'

def __read_json(filename):
    return json.load(open(filename))

def __read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for row in reader:
            yield row

def read(csvfile):
    files = (row[2] + '.json' for row in __read_csv(csvfile))
    jsons = (__read_json(TWEET_DIR + file) for file in files)
    return (json['text'] for json in jsons)
