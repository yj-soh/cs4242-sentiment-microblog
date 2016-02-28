import HTMLParser
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
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
        'out': 'data/generated/development_tweets.txt'
    },
    'testing': {
        'in': 'data/testing.csv',
        'out': 'data/generated/testing_tweets.txt'
    }
}

options = {
    'stopwords': False,
    'force_lowercase': True,
    'trim_repeat_char': True,
    'lemma': False,
    'negation': True,
    # 'escape_special': True,
    'replace_slang': False,
    'no_hash_hashtags': True
}
NEGATION = 'not_'
DATE_FMT = '%a %b %d %H:%M:%S +0000 %Y'

ADVERB_TAGS = ['RB', 'RBR', 'RBS', 'WRB']
NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'WP']
ADJECTIVE_TAGS = ['JJ', 'JJR', 'JJS']
VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

html_parser = HTMLParser.HTMLParser()
pos_tagger = nltk.tag.perceptron.PerceptronTagger()
lemmatizer = WordNetLemmatizer()
stopwords = map(lambda s:str(s), stopwords.words('english'))
slang = reader.read_tsv_map('resources/noslang.csv')
slang = {k:v.lower() for k, v in slang.items()}
'''
escape_words = {
    '\'': '&#39;',
    '\"': '&quot;'
}
'''

re_str_emoji = u'\ud83c[\udf00-\udfff]|\ud83d[\udc00-\ude4f\ude80-\udeff]|[\u2600-\u26FF\u2700-\u27BF]+'

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

re_str_words_meta = r'''
    (?:@[\w]+)                     # Usernames
    |
    (?:\#+[\w]+[\w\'\-]*[\w]+)     # Hashtags
    |
    (?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,}) # URLs
    '''
re_str_words = r'''
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w]+)                      # Words without apostrophes or dashes.
    '''
re_str_words_etc = r'''
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
    .*n't'''

re_emoji = re.compile(re_str_emoji, re.UNICODE)
re_emoticon = re.compile(re_str_emoticon, re.VERBOSE | re.I | re.UNICODE)
re_words = re.compile(re_str_emoticon + '|' + re_str_words_meta + '|' + re_str_words + '|' + re_str_words_etc, \
                      re.VERBOSE | re.I | re.UNICODE)
re_words_only = re.compile(re_str_words, re.VERBOSE | re.I | re.UNICODE)
re_repeat_char = re.compile(r'(.)\1+')
re_negation = re.compile(re_str_negation, re.VERBOSE)
re_clause_punctuation = re.compile('^[.:;!?]$')

def _get_unigrams(text):
    words = re_words.findall(text)
    
    if options['replace_slang']:
        new_words = []
        for w in words:
            if w.lower() in slang:
                new_words.extend(slang[w.lower()].split())
            else:
                new_words.append(w)
        words = new_words
    
    tagged_words = nltk.tag._pos_tag(words, None, pos_tagger)
    
    return tagged_words

def _get_tag_type(tag):
    if tag in ADVERB_TAGS:
        return 'r'
    elif tag in ADJECTIVE_TAGS:
        return 'a'
    elif tag in VERB_TAGS:
        return 'v'
    else:
        return 'n'

def _process_word(word, tag):
    # if is emoticon
    if re_emoticon.search(word):
        return word
    
    ### lower-case operations below ###
    if options['force_lowercase']:
        word = word.lower()
    
    if options['no_hash_hashtags']:
        if word.startswith('#'):
            word = word[1:]
    
    # if is stopword
    if options['stopwords'] and word in stopwords:
        return ''
    
    if options['trim_repeat_char']:
        word = re_repeat_char.sub(r'\1\1', word)
    # todo: English contractions
    if options['lemma']:
        try:
            word = str(lemmatizer.lemmatize(word, _get_tag_type(tag)))
        except UnicodeDecodeError:
            pass
    
    return word

def _negate_range(words, start, end):
    negation = map(lambda w: NEGATION + w if re_words_only.match(w) else w, words[start:end])
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

'''
def _escape_special(str):
    for c in escape_words:
        str = str.replace(c, escape_words[c])
    return str
'''

def extract_emoji(text):
    emoji = re_emoji.findall(text)
    text = re_emoji.sub('', text)
    
    return text, emoji
    

def _parse_text(tweet):
    # extract emoji
    tweet, emoji = extract_emoji(tweet)
    
    # markup normalization
    tweet = html_parser.unescape(tweet)
    tweet = tweet.encode('utf8')
    
    # split into unigrams
    tagged_words = _get_unigrams(tweet)
    
    # process each unigram
    rtweet = []
    
    for tagged_word in tagged_words:
        word, tag = tagged_word
        word = _process_word(word, tag)
        if word:
            rtweet.append((word, tag))
    
    # unpack for words-only processing
    words = [w[0] for w in rtweet]
    tags = [w[1] for w in rtweet]
    
    # after-splitting operations
    if options['negation']:
        words = _handle_negation(words)
    '''
    if options['escape_special']:
        words = map(_escape_special, words)
    '''
    # rtweet = remove punctuation?
    
    # repack into tuples
    rtweet = zip(words, tags)
    
    rtweet.extend((e, 'NN') for e in emoji)
    
    return [w[0] for w in rtweet], rtweet

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
      tagged_unigrams: string[], relevant bits of tweet msg, POS-tagged,
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
        tweet['unigrams'], tweet['tagged_unigrams'] = _parse_text(json['text'])
        
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
        
        f(tweet)

def parse_all_files(new_options=options):
    files = [FILES['training'], FILES['testing'], FILES['development']]
    
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
