import pickle
import os.path
import numpy as np
import csv
import buildfeatures
import tweetparser
from classifier import Classifier

TRAINING_DATA_FILE = 'data/generated/training_data.csv'
TESTING_DATA_FILE = 'data/generated/testing_data.csv'
DEVELOPMENT_DATA_FILE = 'data/generated/development_data.csv'

TRAINING_TWEETS = 'data/generated/training_tweets.txt'
TESTING_TWEETS = 'data/generated/testing_tweets.txt'
DEVELOPMENT_TWEETS = 'data/generated/development_tweets.txt'

TRAINING = 'data/training.csv'
TESTING = 'data/testing.csv'
DEVELOPMENT = 'data/development.csv'

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
        self.parser_options = tweetparser.options

        self.classifier = Classifier()
        if os.path.exists(CLASSIFIER_FILE):
            self.classifier.load_classifier(CLASSIFIER_FILE)
        else:
            self.retrain_classifier()

    def retrain_classifier(self):
        print 'Parsing tweets...'
        tweetparser.parse_all_files(self.parser_options)

        print 'Building features...'
        # build features for training data
        training_labels = read_labels(TRAINING)
        training_tweets = training_tweets = pickle.load(open(TRAINING_TWEETS, 'rb'))
        unigram_features = buildfeatures.build_unigram_feature_dict(training_tweets, training_labels)

        training_data = buildfeatures.get_feature_vectors(training_tweets, unigram_features)

        # save training data
        np.savetxt(TRAINING_DATA_FILE, training_data, delimiter=',')

        # build features for testing data
        testing_tweets = pickle.load(open(TESTING_TWEETS, 'rb'))
        testing_data = buildfeatures.get_feature_vectors(testing_tweets, unigram_features)
        np.savetxt(TESTING_DATA_FILE, testing_data, delimiter=',')

        # build features for development data
        development_tweets = pickle.load(open(DEVELOPMENT_TWEETS, 'rb'))
        development_data = buildfeatures.get_feature_vectors(development_tweets, unigram_features)
        np.savetxt(DEVELOPMENT_DATA_FILE, development_data, delimiter=',')

        # save unigram features processed
        pickle.dump(unigram_features, open(UNIGRAM_FEATURES_FILE, 'wb'), -1)

        print 'Training classifier...'
        self.classifier = Classifier()
        self.classifier.train(training_data, training_labels)
        self.classifier.save_classifier(CLASSIFIER_FILE)

    def classify_test_tweets(self):
        testing_tweets = pickle.load(open(TESTING_TWEETS, 'rb'))
        testing_data = np.loadtxt(TESTING_DATA_FILE, delimiter=',')
        testing_labels = read_labels(TESTING)

        print 'Predicting labels...'
        print 'Results: ' + str(self.classifier.predict_testing_data(testing_tweets, testing_data, testing_labels, RESULTS_FILE))

    def classify_development_tweets(self):
        development_tweets = pickle.load(open(DEVELOPMENT_TWEETS, 'rb'))
        development_data = np.loadtxt(DEVELOPMENT_DATA_FILE, delimiter=',')
        development_labels = read_labels(DEVELOPMENT)

        print 'Predicting labels...'
        print 'Results: ' + str(self.classifier.predict_testing_data(development_tweets, development_data, development_labels, RESULTS_FILE))

    def adjust_parser(self):
        length = len(self.parser_options)
        option = 0
        while not option == length + 1:
            print 'Which parser switch do you want to flip?'
            switches = {}

            for i, (opt, val) in enumerate(self.parser_options.items()):
                switches[i + 1] = opt
                print str(i + 1) + '. ' + opt + ':' + (' ' * (24 - len(opt))) + str(val)
            print str(length + 1) + '. Back to main menu'

            option = input('Answer: ')
            if option > 0 and option < length + 1:
                opt = switches[option]
                self.parser_options[opt] = not self.parser_options[opt]

if __name__ == '__main__':
    print 'Loading classifier...'
    sentimentAnalyzer = SentimentAnalyzer()

    option = 0
    while option != 6:
        option = input('What do you want to do?\n1. Retrain Classifier\n2. Classify Test Tweets\n3. Classify Development Tweets\n4. Adjust Parser Options\n5. Classify Custom Tweets\n6. Goodbye\nAnswer: ')

        if option == 1:
            print 'Please wait...'
            sentimentAnalyzer.retrain_classifier()
            print 'Classifier trained!'
        elif option == 2:
            print 'Please wait...'
            sentimentAnalyzer.classify_test_tweets()
            print 'See labels at: ' + RESULTS_FILE
        elif option == 3:
            print 'Please wait...'
            sentimentAnalyzer.classify_development_tweets()
            print 'See labels at: ' + RESULTS_FILE
        elif option == 4:
            sentimentAnalyzer.adjust_parser()
            print 'Parser options updated!'
