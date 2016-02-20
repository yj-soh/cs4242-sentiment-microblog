import pickle
import os.path
import numpy as np
import csv
import buildfeatures
import tweetparser
from classifier import Classifier

TRAINING_DATA_FILE = 'data/generated/training_data.csv'
TESTING_DATA_FILE = 'data/generated/testing_data.csv'
TRAINING_TWEETS = 'data/generated/training_tweets.txt'
TESTING_TWEETS = 'data/generated/testing_tweets.txt'
TRAINING = 'data/training.csv'
TESTING = 'data/testing.csv'
CLASSIFIER_FILE = 'data/generated/classifier.txt'
UNIGRAM_FEATURES_FILE = 'data/generated/unigram_features.txt'
RESULTS_FILE = 'results.csv'

def read_labels(filename):
    labels = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for row in reader:
            labels.append(row[1])

    return labels

class SentimentAnalyzer:
    def __init__(self):
        self.classifier = Classifier()
        if os.path.exists(CLASSIFIER_FILE):
            self.classifier.load_classifier(CLASSIFIER_FILE)
        else:
            self.retrain_classifier()

    def retrain_classifier(self):
        print 'Parsing tweets...'
        tweetparser.parse_all_files()

        print 'Building features...'
        # build features for training data
        training_tweets = training_tweets = pickle.load(open(TRAINING_TWEETS, 'rb'))
        unigram_features, training_data = buildfeatures.get_feature_vectors(training_tweets, dict())

        # save training data
        np.savetxt(TRAINING_DATA_FILE, training_data, delimiter=',')

        # build features for testing data
        testing_tweets = pickle.load(open(TESTING_TWEETS, 'rb'))
        _, testing_data = buildfeatures.get_feature_vectors(testing_tweets, unigram_features)
        np.savetxt(TESTING_DATA_FILE, testing_data, delimiter=',')

        # save unigram features processed
        pickle.dump(unigram_features, open(UNIGRAM_FEATURES_FILE, 'wb'), -1)

        training_labels = read_labels(TRAINING)

        print 'Training classifier...'
        self.classifier.train(training_data, training_labels)
        self.classifier.save_classifier(CLASSIFIER_FILE)

    def classify_test_tweets(self):
        testing_data = np.loadtxt(TESTING_DATA_FILE, delimiter=',')
        testing_labels = read_labels(TESTING)

        print 'Predicting labels...'
        print 'Results: ' + str(self.classifier.predict_testing_data(testing_data, testing_labels, RESULTS_FILE))

if __name__ == '__main__':
    print 'Loading classifier...'
    sentimentAnalyzer = SentimentAnalyzer()

    option = 0
    while option != 4:
        option = input('What do you want to do?\n1. Retrain Classifier\n2. Classify Test Tweets\n3. Classify Custom Tweets\n4. Goodbye\nAnswer: ')

        if option == 1:
            print 'Please wait...'
            sentimentAnalyzer.retrain_classifier()
            print 'Classifier trained!'
        elif option == 2:
            print 'Please wait...'
            sentimentAnalyzer.classify_test_tweets()
            print 'See labels at: ' + RESULTS_FILE