import nltk
import csv
import numpy as np

from nltk.tag import pos_tag

NEGATION = 'not_'

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

    @staticmethod
    def tag(words):
        tagged_words = nltk.pos_tag(words)
        result = list()

        for tagged_word in tagged_words: # tagged_word = (word, tag)
            result.append(tagged_word[0] + '/' + tagged_word[1])

        return result

def get_feature_vectors(tweets):
    sentiment_scorer = SentimentScorer()
    sentiment_scores = np.zeros((len(tweets), 1))

    unigram_feature_dict = dict() # maps feature to first time it is seen
    count_features = 0

    tagged_tweets = list()

    for index, tweet in enumerate(tweets):
        # put sentiment score in first column
        sentiment_scores[index, 0] = sentiment_scorer.get_sentiment_score(tweet)
        tagged_tweet = POSTagger.tag(tweet)

        for word in tagged_tweet:
            if (word not in unigram_feature_dict):
                count_features += 1
                unigram_feature_dict[word] = count_features - 1

        tagged_tweets.append(tagged_tweet)

    unigram_feature_vectors = np.zeros((len(tweets), len(unigram_feature_dict)))
    
    for index, tweet in enumerate(tagged_tweets):
        for word in tweet:
            unigram_feature_vectors[index, unigram_feature_dict[word]] = 1

    return np.concatenate((sentiment_scores, unigram_feature_vectors), axis=1)
            
if __name__ == '__main__':
    tweets = [['hello', 'there', 'happy', ':)', 'puppy'], ['hello', 'there', 'happy', ':)', 'puppy'], ['sup', 'not', 'a', 'tweet']]
    print get_feature_vectors(tweets)