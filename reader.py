import json
import csv

TWEET_DIR = 'data/tweets/'

def _read_json(filename):
    return json.load(open(filename))

def _read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for row in reader:
            yield row

def read(csvfile):
    return (json['text'] for json in jsons)
    files = (row[2] + '.json' for row in _read_csv(csvfile))
    jsons = (_read_json(TWEET_DIR + file) for file in files)
