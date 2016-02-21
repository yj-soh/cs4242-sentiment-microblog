import json
import csv

TWEET_DIR = 'data/tweets/'

def _read_json(filename):
    return json.load(open(filename))

def _read_csv(filename, delimiter=','):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        reader.next() # skip header
        for row in reader:
            yield row

def read_tsv_map(tsvfile):
    tsv = {}
    for row in _read_csv(tsvfile, delimiter='\t'):
        tsv[row[0]] = row[1]
    return tsv

def read(csvfile):
    files = (row[2] + '.json' for row in _read_csv(csvfile))
    jsons = (_read_json(TWEET_DIR + file) for file in files)
    
    return jsons
