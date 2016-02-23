import numpy as np
import pickle
import csv
import os.path
from sklearn import metrics
# classifiers
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

TRAINING_DATA_FILE = 'data/generated/training_data.csv'
TESTING_DATA_FILE = 'data/generated/testing_data.csv'
TRAINING = 'data/training.csv'
DEVELOPMENT = 'data/development.csv'
TESTING = 'data/testing.csv'
CLASSIFIER_FILE = 'data/generated/classifier.txt'
RESULTS_FILE = 'data/generated/results.csv'

class Classifier:
    def __init__(self):
        
        # {'recall': 0.66512708133668053, 'precision': 0.71939463882993837, 'F1': 0.68568062271511643}
        self.classifier = LinearSVC(class_weight='auto')
        
        # {'recall': 0.60011728328447156, 'precision': 0.60452209108905675, 'F1': 0.60098957336006997}
        # self.classifier = GaussianNB()

        # {'recall': 0.52459268280775861, 'precision': 0.74282309921659195, 'F1': 0.54471600331876879}
        # self.classifier = RandomForestClassifier()

        # self.classifier = KNeighborsClassifier(10) # lousy one

    def train(self, training_data, training_labels):
        self.classifier.fit(training_data, training_labels)

    def save_classifier(self, filename):
        pickle.dump(self.classifier, open(filename, 'wb'))

    def load_classifier(self, filename):
        self.classifier = pickle.load(open(filename, 'rb'))

    # saves overall results in results_file. Returns [recall, precision, F1]
    def predict_testing_data(self, testing_tweets, testing_data, testing_topics, testing_labels, results_file):
        result_labels = self.classifier.predict(testing_data)
        csv_writer = csv.writer(open(results_file, 'wb'))
        csv_writer.writerow(['Topic', 'Sentiment', 'TwitterText'])
        for index, label in enumerate(result_labels):
            csv_writer.writerow([testing_topics[index], label, testing_tweets[index]['text'].encode('utf8')])

        accuracy = metrics.accuracy_score(testing_labels, result_labels)
        precision = metrics.precision_score(testing_labels, result_labels, average='macro')
        recall = metrics.recall_score(testing_labels, result_labels, average='macro')
        f1 = metrics.f1_score(testing_labels, result_labels, average='macro')

        return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'F1': f1}

    def predict(self, testing_data):
        result_labels = self.classifier.predict(testing_data)
        return result_labels
