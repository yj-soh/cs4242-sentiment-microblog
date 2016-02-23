import nltk
import csv
import reader
import numpy as np
import os.path
import pickle

from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVR
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier

NEGATION = 'not_'
ADVERB_TAGS = ['RB', 'RBR', 'RBS', 'WRB']
NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'WP']
ADJECTIVE_TAGS = ['JJ', 'JJR', 'JJS']
VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

UNIGRAM_FEATURES_FILE = 'data/generated/unigram_features.txt'
TRAINING_TWEETS_FILE = 'data/generated/training_tweets.txt'
DEVELOPMENT_TWEETS_FILE = 'data/generated/development_tweets.txt'
TESTING_TWEETS_FILE = 'data/generated/testing_tweets.txt'

class SentimentScorer:

    def __init__(self):
        def create_dictionary_from_csv(csv_filename, delimiter=','):
            dictionary = {}
            csv_reader = csv.reader(open(csv_filename, 'r'), delimiter=delimiter)
            for row in csv_reader:
                key = row[0]
                sentiment_score = float(row[1])
                dictionary[key] = sentiment_score
                dictionary[NEGATION + key] = -sentiment_score # add negation to dictionary as well

            return dictionary

        self.lemmatizer = WordNetLemmatizer()

        # initialize sentiment lexicons
        self.neg_lexicon = set()
        self.pos_lexicon = set()

        txt_file = open('data/lexicon/neg.txt', 'r')
        for line in txt_file:
            word = line.strip()
            try:
                word = self.lemmatizer.lemmatize(word)
            except UnicodeDecodeError:
                pass
            self.neg_lexicon.add(word)
            self.pos_lexicon.add(NEGATION + word) # add negation to other lexicon set

        txt_file = open('data/lexicon/pos.txt', 'r')
        for line in txt_file:
            word = line.strip()
            try:
                word = self.lemmatizer.lemmatize(word)
            except UnicodeDecodeError:
                pass
            self.pos_lexicon.add(word)
            self.neg_lexicon.add(NEGATION + word)

        txt_file.close()

        # initialize emoji and emoticon lexicons
        self.emoji_lexicon = {}
        for emoji, sentiment in create_dictionary_from_csv('resources/emoji-lexicon.csv').items():
            try:
                self.emoji_lexicon[emoji.decode('unicode_escape')] = sentiment
            except UnicodeDecodeError:
                self.emoji_lexicon[emoji] = sentiment
        self.emoticon_lexicon = create_dictionary_from_csv('resources/emoticon-lexicon.txt', delimiter='\t')

    def get_sentiment_score(self, words):
        sentiment_score = 0
        for word in words:
            if (word in self.neg_lexicon):
                sentiment_score -= 1
            elif (word in self.pos_lexicon):
                sentiment_score += 1
            elif (word in self.emoji_lexicon):
                sentiment_score += self.emoji_lexicon[word]
            elif (word in self.emoticon_lexicon):
                sentiment_score += self.emoticon_lexicon[word]

        return sentiment_score

class POSTagger:

    def __init__(self):
        self.tagger = nltk.tag.perceptron.PerceptronTagger()

    def tag(self, words): # returns count of each tag
        def pos_tag(tokens):
            tagset = None
            return nltk.tag._pos_tag(tokens, tagset, self.tagger)

        def initialize_tag_count():
            tag_count = {}
            for tag in ADVERB_TAGS + NOUN_TAGS + VERB_TAGS + ADJECTIVE_TAGS:
                tag_count[tag] = 0
            tag_count['other'] = 0

            return tag_count

        tagged_words = pos_tag(words)
        total_words = len(words)

        tag_count = initialize_tag_count()

        for (word, tag) in tagged_words:
            if (tag in ADVERB_TAGS + NOUN_TAGS + VERB_TAGS + ADJECTIVE_TAGS):
                tag_count[tag] += (1.0 / total_words)
            else:
                tag_count['other'] += (1.0 / total_words)

        return tag_count

# build unigram feature dictionary, format: (key = word/feature, value = index of feature)
# do feature selection
def build_unigram_feature_dict(tweets, tweet_labels):
    unigram_feature_dict = dict()
    all_unigram_features = []
    count_features = 0
    # process unigram features
    for index, tweet in enumerate(tweets):
        for word in tweet['unigrams']:
            if (word not in unigram_feature_dict):
                count_features += 1
                unigram_feature_dict[word] = count_features - 1
                all_unigram_features.append(word)

    unigram_feature_vectors = np.zeros((len(tweets), len(unigram_feature_dict)))

    for index, tweet in enumerate(tweets):
        tweet_unigrams = tweet['unigrams']
        for word in tweet_unigrams:
            if (word in unigram_feature_dict):
                unigram_feature_vectors[index, unigram_feature_dict[word]] += 1

    # select features using chi2
    select = SelectPercentile(chi2, percentile=80)
    select.fit(unigram_feature_vectors, tweet_labels)
    selected_features = select.get_support()

    # select features using variance (not that good)
    # select = VarianceThreshold(threshold=0.0001)
    # select.fit(unigram_feature_vectors, tweet_labels)
    # selected_features = select.get_support()

    # select features using ANOVA F value
    # select = SelectFpr(f_classif, alpha=0.75)
    # select.fit(unigram_feature_vectors, tweet_labels)
    # selected_features = select.get_support()

    # select features using ExtraTreesClassifier (can be best but not deterministic)
    # np.random.seed(3597475650) # set seed so that results always the same
    # forest = ExtraTreesClassifier()
    # forest.fit(unigram_feature_vectors, tweet_labels)
    # index = np.arange(0, unigram_feature_vectors.shape[1]) # create an index array, with the number of features
    # threshold_index = len(forest.feature_importances_) * 0.65
    # threshold = np.partition(forest.feature_importances_, threshold_index)[threshold_index]
    # # threshold = np.mean(np.percentile(forest.feature_importances_, 70, axis=0)) # anyhow
    # selected_features_indexes = index[forest.feature_importances_ > threshold]
    # selected_features = np.zeros((unigram_feature_vectors.shape[1]), dtype=bool)
    # selected_features[selected_features_indexes] = True

    # remove unwanted features
    for index, is_feature_selected in reversed(list(enumerate(selected_features))):
        if (not is_feature_selected):
            del all_unigram_features[index]

    # create new dictionary based on selected features
    unigram_feature_dict = dict()
    for index, unigram in enumerate(all_unigram_features):
        unigram_feature_dict[unigram] = index
    
    return unigram_feature_dict

# unigram_feature_dict in this format: (key = word/feature, value = index of feature)
def get_feature_vectors(tweets, unigram_feature_dict):
    pos_tagger = POSTagger()
    sentiment_scorer = SentimentScorer()
    sentiment_scores = np.zeros((len(tweets), 1))
    social_features = np.zeros((len(tweets), 4)) # rt_count, fav_count, mention_count, friend_count

    tag_count_features = list()

    unigram_feature_vectors = np.zeros((len(tweets), len(unigram_feature_dict)))
    
    for index, tweet in enumerate(tweets):
        tweet_unigrams = tweet['unigrams']
        # put sentiment score in first column
        sentiment_scores[index, 0] = sentiment_scorer.get_sentiment_score(tweet_unigrams)

        # put date time in first column
        # date_time_values[index, 0] = tweet['datetime']/60/60/24 # just get the day

        # social features
        social_features[index, 0] = tweet['rt_count']
        # social_features[index, 1] = tweet['fav_count']
        social_features[index, 2] = len(tweet['users'])

        tag_count = pos_tagger.tag(tweet_unigrams)

        tag_count_list = []
        for tag, count in tag_count.iteritems():
            tag_count_list.append(count)

        tag_count_features.append(tag_count_list) # add tag count as feature

        for word in tweet_unigrams:
            if (word in unigram_feature_dict):
                unigram_feature_vectors[index, unigram_feature_dict[word]] = 1 # term presence

    return np.concatenate((sentiment_scores, social_features, unigram_feature_vectors, np.array(tag_count_features)), axis=1)
            
if __name__ == '__main__':
    tweets = [
        {'users': [u'17739746', u'380749300', None], 'rt_count': 0, 'text': u'Good support fm Kevin @apple #Bellevue store 4 biz customers TY!', 'datetime': 1318668623.0, 'unigrams': ['good', 'support', 'kevin', '@apple', 'bellevue', 'store', 'customer', '!', 'fuck', 'business', 'thank'], 'fav_count': 0}
    ]
    unigram_features = pickle.load(open(UNIGRAM_FEATURES_FILE, 'rb'))
