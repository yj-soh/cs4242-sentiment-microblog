import HTMLParser
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import re

import reader

FILES = {
    'training': {
        'in': 'data/training.csv',
        'out': 'data/generated/training_tweets.txt'
    },
    'development': {
        'in': 'data/development.csv',
        'out': 'data/generated/development_weets.txt'
    },
    'testing': {
        'in': 'data/testing.csv',
        'out': 'data/generated/testing_tweets.txt'
    }
}

NEGATION = 'not_'

html_parser = HTMLParser.HTMLParser()
lemmatizer = WordNetLemmatizer()
stopwords = map(lambda s:str(s), stopwords.words('english'))
escape_words = {
    '\'': '&#39;',
    '\"': '&quot;'
}

re_str_emoticon = r'''
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )'''

re_str_words = r'''
    (?:@[\w]+)                     # Usernames
    |
    (?:\#+[\w]+[\w\'\-]*[\w]+)     # Hashtags
    |
    (?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,}) # URLs
    |
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w]+)                      # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    '''

re_str_negation = r'''
    (?:
        ^(?:never|no|nothing|nowhere|noone|none|not|
            havent|hasnt|hadnt|cant|couldnt|shouldnt|
            wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
        )$
    )
    |
    n't'''

re_emoticon = re.compile(re_str_emoticon, re.VERBOSE | re.I | re.UNICODE)
re_words = re.compile(re_str_emoticon + '|' + re_str_words, \
                      re.VERBOSE | re.I | re.UNICODE)
re_repeat_char = re.compile(r'(.)\1+')
re_negation = re.compile(re_str_negation, re.VERBOSE)
re_clause_punctuation = re.compile('^[.:;!?]$')

def _process_word(word):
    # if is emoticon
    if re_emoticon.search(word):
        return [word]
    
    # if is stopword
    if word in stopwords:
        return ['']
    
    # else process word
    word = word.lower()
    word = re_repeat_char.sub(r'\1\1', word)
    # contractions
    # slang
    try:
        word = str(lemmatizer.lemmatize(word))
    except UnicodeDecodeError:
        pass
    
    return [word]

def _negate_range(words, start, end):
    negation = map(lambda w: NEGATION + w, words[start:end])
    return words[:start] + negation + words[end:]

def _handle_negation(words):
    negations = []
    punctuations = []
    is_negation_next = True
    
    # alternates indices between negation and punctuation
    for idx, word in enumerate(words):
        if is_negation_next and re_negation.match(word):
            negations.append(idx + 1)
            is_negation_next = False
        if not is_negation_next and re_clause_punctuation.match(word):
            punctuations.append(idx)
            is_negation_next = True
    # negates everything ahead if no punctuation found
    punctuations.append(len(words))
    
    if not negations:
        return words
    
    negation_ranges = zip(negations, punctuations)
    
    for negation_range in negation_ranges:
        start, end = negation_range
        words = _negate_range(words, start, end)
    
    return words

def _escape_special(str):
    for c in escape_words:
        str = str.replace(c, escape_words[c])
    return str

def parse_tweets(tweets_csv, f):
    for tweet in reader.read(tweets_csv):
        # markup normalization
        tweet = html_parser.unescape(tweet)
        tweet = tweet.encode('utf8')
        
        words = re_words.findall(tweet)
        
        rtweet = []
        for word in words:
            result = process_word(word)
            if isinstance(result, list):
                rtweet.extend(result)
            else:
                rtweet.append(result)
        rtweet = filter(None, rtweet) # remove empty strings

        rtweet = handle_negation(rtweet)
        rtweet = map(escape_special, rtweet)
        # rtweet = remove punctuation?
        
        f(rtweet)

def parse_all_files():
    files = [FILES['training'], FILES['testing']]
    
    for type in files:
        tweets_csv = type['in']
    
        # toss everything into memory; should be fine due to data's size
        text_arrs = []
        def collect(text_arr):
            text_arrs.append(text_arr)
        _parse_tweets(tweets_csv, collect)
        
        f = open(type['out'], 'wb')
        pickle.dump(text_arrs, f, -1)
        f.close()

if __name__ == '__main__':
        parse_all_files()
        # text_arrs format: [[word, ...], [word, ...], ...]
        
        ''' # Reading the file:
        f = open('out.txt', rb')
        text_arrs = pickle.load(f)
        f.close()
        '''
