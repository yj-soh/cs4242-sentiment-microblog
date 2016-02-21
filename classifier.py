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
        # {'recall': 0.66452140497559475, 'precision': 0.69705655218910723, 'F1': 0.67857497717157655}
        self.classifier = LinearSVC(class_weight='auto')

        # {'recall': 0.60981487119929356, 'precision': 0.61804837692264081, 'F1': 0.61289175026667586}
        # self.classifier = GaussianNB()
        
        # {'recall': 0.52133906999012491, 'precision': 0.74365097295907634, 'F1': 0.54587294497124295}
        # self.classifier = RandomForestClassifier()

        # self.classifier = KNeighborsClassifier(20) # lousy one

    def train(self, training_data, training_labels):
        self.classifier.fit(training_data, training_labels)

    def save_classifier(self, filename):
        pickle.dump(self.classifier, open(filename, 'wb'))

    def load_classifier(self, filename):
        self.classifier = pickle.load(open(filename, 'rb'))

    # saves overall results in results_file. Returns [recall, precision, F1]
    def predict_testing_data(self, testing_data, testing_labels, results_file):
        result_labels = self.classifier.predict(testing_data)
        csv_writer = csv.writer(open(results_file, 'wb'))
        for label in result_labels:
            csv_writer.writerow([label])

        precision = metrics.precision_score(testing_labels, result_labels, average='macro')
        recall = metrics.recall_score(testing_labels, result_labels, average='macro')
        f1 = metrics.f1_score(testing_labels, result_labels, average='macro')

        return {'recall': recall, 'precision': precision, 'F1': f1}

    def predict(self, testing_data):
        result_labels = self.classifier.predict(testing_data)
        return result_labels


if __name__ == '__main__':
    def read_labels(filename):
        labels = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            reader.next() # skip header
            for row in reader:
                labels.append(row[1])

        return labels

    classifier = Classifier()

    # if classifier already trained and loaded
    if os.path.exists(CLASSIFIER_FILE):
        classifier.load_classifier(CLASSIFIER_FILE)
    else:
        training_data = np.loadtxt(TRAINING_DATA_FILE, delimiter=',')
        training_labels = read_labels(TRAINING)
        classifier.train(training_data, training_labels)

    testing_data = np.loadtxt(TESTING_DATA_FILE, delimiter=',')
    testing_labels = read_labels(TESTING)
    print classifier.predict_testing_data(testing_data, testing_labels, RESULTS_FILE)

    classifier.save_classifier(CLASSIFIER_FILE)
