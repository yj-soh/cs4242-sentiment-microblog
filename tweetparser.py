import HTMLParser
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import re
from time import mktime, strptime

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

options = {
    'stopwords': True,
    'force_lowercase': True,
    'trim_repeat_char': True,
    'lemma': True,
    'negation': True,
    'escape_special': True
}
NEGATION = 'not_'
DATE_FMT = '%a %b %d %H:%M:%S +0000 %Y'

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
    if options['stopwords'] and word in stopwords:
        return ['']
    
    # else process word
    if options['force_lowercase']:
        word = word.lower()
    if options['trim_repeat_char']:
        word = re_repeat_char.sub(r'\1\1', word)
    # contractions
    # slang
    if options['lemma']:
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

def _parse_text(tweet):
    # markup normalization
    tweet = html_parser.unescape(tweet)
    tweet = tweet.encode('utf8')
    
    # split into unigrams
    words = re_words.findall(tweet)
    
    # process each unigram
    rtweet = []
    for word in words:
        rtweet.extend(_process_word(word))
    
    # remove empty strings
    rtweet = filter(None, rtweet)

    # after-splitting operations
    if options['negation']:
        rtweet = _handle_negation(rtweet)
    if options['escape_special']:
        rtweet = map(_escape_special, rtweet)
    # rtweet = remove punctuation?
    
    return rtweet

def _parse_datetime(date_str):
    return mktime(strptime(date_str, DATE_FMT))

def _append_if_exists(src, dst, key):
    try:
        dst[key].append(src)
    except:
        pass

def _extend_if_exists(src, src_key, dst, dst_key):
    try:
        dst[dst_key].extend(d[src_key] for d in src)
    except KeyError:
        pass

def _parse_tweets(tweets_csv, f):
    '''
    format of each tweet: {
      text: string, original text of tweet msg,
      unigrams: string[], relevant bits of tweet msg,
      datetime: float, created date of tweet in Unix time,
      users: string[], user ids of relevant users,
      rt_count: int, no. of times retweeted,
      fav_count: int, no. of times favorited
    }
    '''
    for json in reader.read(tweets_csv):
        tweet = {}
        
        # text
        tweet['text'] = json['text']
        tweet['unigrams'] = _parse_text(json['text'])
        
        # datetime
        tweet['datetime'] = _parse_datetime(json['created_at'])
        
        # user ids
        tweet['users'] = []
        _append_if_exists(json['user']['id_str'], tweet, 'users')
        _extend_if_exists(json['entities']['user_mentions'], 'id_str', tweet, 'users')
        _append_if_exists(json['in_reply_to_user_id_str'], tweet, 'users')
        
        # counts
        tweet['rt_count'] = json['retweet_count']
        tweet['fav_count'] = json['favorite_count']
        print str(tweet['rt_count']) + ' ' + str(tweet['fav_count'])
        
        f(tweet)

def parse_all_files(new_options=options):
    files = [FILES['training'], FILES['testing']]
    
    options = new_options
    
    for type in files:
        tweets_csv = type['in']
    
        # toss everything into memory; should be fine due to data's size
        tweets = []
        def collect(tweet):
            tweets.append(tweet)
        _parse_tweets(tweets_csv, collect)
        
        f = open(type['out'], 'wb')
        pickle.dump(tweets, f, -1)
        f.close()

if __name__ == '__main__':
        parse_all_files()
        # text_arrs format: [[word, ...], [word, ...], ...]
        
        ''' # Reading the file:
        f = open('out.txt', rb')
        text_arrs = pickle.load(f)
        f.close()
        '''
