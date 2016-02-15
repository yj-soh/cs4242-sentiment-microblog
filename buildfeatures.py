import nltk
import csv
import reader
import numpy as np
import os.path
import pickle

from nltk.tag import pos_tag

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

        # initialize sentiment lexicons
        self.neg_lexicon = set()
        self.pos_lexicon = set()

        txt_file = open('data/lexicon/neg.txt', 'r')
        for line in txt_file:
            word = line.strip()
            self.neg_lexicon.add(word)
            self.pos_lexicon.add(NEGATION + word) # add negation to other lexicon set

        txt_file = open('data/lexicon/pos.txt', 'r')
        for line in txt_file:
            word = line.strip()
            self.pos_lexicon.add(word)
            self.neg_lexicon.add(NEGATION + word)

        txt_file.close()

        # initialize emoji and emoticon lexicons
        self.emoji_lexicon = create_dictionary_from_csv('resources/emoji-lexicon.csv')
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
        print tagged_words

        tag_count = initialize_tag_count()

        for (word, tag) in tagged_words:
            if (tag in ADVERB_TAGS + NOUN_TAGS + VERB_TAGS + ADJECTIVE_TAGS):
                tag_count[tag] += 1
            else:
                tag_count['other'] += 1

        return tag_count

# unigram_feature_dict in this format: (key = word/feature, value = index of feature)
def get_feature_vectors(tweets, unigram_feature_dict = dict()):
    pos_tagger = POSTagger()
    sentiment_scorer = SentimentScorer()
    sentiment_scores = np.zeros((len(tweets), 1))

    tag_count_features = list()
    count_features = 0

    if (len(unigram_feature_dict) == 0):
        # process unigram features
        for index, tweet in enumerate(tweets):
            for word in tweet:
                if (word not in unigram_feature_dict):
                    count_features += 1
                    unigram_feature_dict[word] = count_features - 1

    unigram_feature_vectors = np.zeros((len(tweets), len(unigram_feature_dict)))
    
    for index, tweet in enumerate(tweets):
        # put sentiment score in first column
        sentiment_scores[index, 0] = sentiment_scorer.get_sentiment_score(tweet)
        tag_count = pos_tagger.tag(tweet)

        tag_count_list = []
        for tag, count in tag_count.iteritems():
            tag_count_list.append(count)

        tag_count_features.append(tag_count_list) # add tag count as feature

        for word in tweet:
            if (word in unigram_feature_dict):
                unigram_feature_vectors[index, unigram_feature_dict[word]] = 1 # term presence

    return unigram_feature_dict, np.concatenate((sentiment_scores, unigram_feature_vectors, np.array(tag_count_features)), axis=1)
            
if __name__ == '__main__':
    # if features already processed, don't have to do it again
    if os.path.exists(UNIGRAM_FEATURES_FILE):
        unigram_features = pickle.load(open(UNIGRAM_FEATURES_FILE, 'rb'))
    else:
        unigram_features = dict()

    # process training features
    training_tweets = pickle.load(open(TRAINING_TWEETS_FILE, 'rb'))
    unigram_features, training_data = get_feature_vectors(training_tweets, unigram_features)

    if not os.path.exists(UNIGRAM_FEATURES_FILE):
        # save unigram features processed
        pickle.dump(unigram_features, open(UNIGRAM_FEATURES_FILE, 'wb'), -1)

    np.savetxt('data/generated/training_data.csv', training_data, delimiter=',')

    # process testing features
    testing_tweets = pickle.load(open(TESTING_TWEETS_FILE, 'rb'))
    _, testing_data = get_feature_vectors(testing_tweets, unigram_features)
    np.savetxt('data/generated/testing_data.csv', testing_data, delimiter=',')