import pickle
import os.path
import numpy as np
import csv
import buildfeatures
import tweetparser
from classifier import Classifier
from sklearn import cross_validation

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

def read_topics(filename):
    topics = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        reader.next() # skip header
        for row in reader:
            topics.append(row[0])

    return topics

class SentimentAnalyzer:
    def __init__(self):
        self.parser_options = tweetparser.options

        self.classifier = Classifier()
        if os.path.exists(CLASSIFIER_FILE):
            self.classifier.load_classifier(CLASSIFIER_FILE)
        else:
            self.retrain_classifier()

    def rebuild_features(self):
        print 'Parsing tweets...'
        tweetparser.parse_all_files(self.parser_options)

        print 'Building features...'
        # build features for training data
        training_labels = read_labels(TRAINING)
        training_tweets = pickle.load(open(TRAINING_TWEETS, 'rb'))
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


    def retrain_classifier(self):
        if not os.path.exists(TRAINING_DATA_FILE):
            self.rebuild_features()
        training_data = np.loadtxt(TRAINING_DATA_FILE, delimiter=',')
        training_labels = read_labels(TRAINING)

        print 'Training classifier...'
        self.classifier = Classifier()
        self.classifier.train(training_data, training_labels)
        self.classifier.save_classifier(CLASSIFIER_FILE)

    def classify_test_tweets(self):
        testing_tweets = pickle.load(open(TESTING_TWEETS, 'rb'))
        testing_data = np.loadtxt(TESTING_DATA_FILE, delimiter=',')
        testing_labels = read_labels(TESTING)
        testing_topics = read_topics(TESTING)

        print 'Predicting labels...'
        print 'Testing Results: ' + str(self.classifier.predict_testing_data(testing_tweets, testing_data, testing_topics, testing_labels, RESULTS_FILE))

    def classify_development_tweets(self):
        development_tweets = pickle.load(open(DEVELOPMENT_TWEETS, 'rb'))
        development_data = np.loadtxt(DEVELOPMENT_DATA_FILE, delimiter=',')
        development_labels = read_labels(DEVELOPMENT)
        development_topics = read_topics(DEVELOPMENT)

        print 'Predicting labels...'
        print 'Development Results: ' + str(self.classifier.predict_testing_data(development_tweets, development_data, development_topics, development_labels, RESULTS_FILE))

    def classify_custom_tweets(self, custom_filename):
        if not os.path.exists(custom_filename):
            print 'The file ' + custom_filename + ' does not exist.'
            return

        try:
            print 'Parsing tweets...'
            custom_tweets = []
            def collect(tweet):
                custom_tweets.append(tweet)
            tweetparser._parse_tweets(custom_filename, collect)
            labels = read_labels(custom_filename)
            topics = read_topics(custom_filename)

            print 'Building features...'
            unigram_features = pickle.load(open(UNIGRAM_FEATURES_FILE, 'rb'))
            data = buildfeatures.get_feature_vectors(custom_tweets, unigram_features)

            print 'Predicting labels...'
            labels = read_labels(custom_filename)
            topics = read_topics(custom_filename)
            print 'Results: ' + str(self.classifier.predict_testing_data(custom_tweets, data, topics, labels, RESULTS_FILE))
            print 'See labels at: ' + RESULTS_FILE
        except:
            print 'Something went wrong. File may be in wrong format.'

    def cross_validation(self):
        training_data = np.loadtxt(TRAINING_DATA_FILE, delimiter=',')
        training_labels = read_labels(TRAINING)

        raw_classifier = self.classifier.get_classifier()
        kf_total = cross_validation.KFold(len(training_labels), n_folds=10, shuffle=True, random_state=4)

        print 'Average F1-Score: ' + str(np.average(cross_validation.cross_val_score(raw_classifier, training_data, training_labels, cv=kf_total, n_jobs=1, scoring='f1_weighted')))

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
    while option != 8:
        option = input('What do you want to do?\n1. Rebuild Features\n2. Retrain Classifier\n3. Classify Test Tweets\n4. Classify Development Tweets\n5. Adjust Parser Options\n6. Classify Custom Tweets\n7. Cross Validate Training Tweets\n8. Goodbye\nAnswer: ')

        if option == 1:
            print 'Please wait...'
            sentimentAnalyzer.rebuild_features()
            print 'Features built!'
        elif option == 2:
            print 'Please wait...'
            sentimentAnalyzer.retrain_classifier()
            print 'Classifier trained!'
        elif option == 3:
            print 'Please wait...'
            sentimentAnalyzer.classify_test_tweets()
            print 'See labels at: ' + RESULTS_FILE
        elif option == 4:
            print 'Please wait...'
            sentimentAnalyzer.classify_development_tweets()
            print 'See labels at: ' + RESULTS_FILE
        elif option == 5:
            sentimentAnalyzer.adjust_parser()
            print 'Parser options updated!'
        elif option == 6:
            # sample format: data/testing.csv
            custom_file = raw_input('Input path to custom tweets file: ')
            print 'Please wait...'
            sentimentAnalyzer.classify_custom_tweets(custom_file)
        elif option ==7:
            print 'Please wait...'
            sentimentAnalyzer.cross_validation()

