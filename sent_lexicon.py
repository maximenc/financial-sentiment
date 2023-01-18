#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import pandas as pd

path_lexicon = "/data/lexicons/l1.csv"

df_lexicon = pd.read_csv(path_lexicon)
df_lexicon["gram_type"] = df_lexicon.keyword.apply(lambda x: len(x.split()))


#--------------word list l1 & l2 :
stopwords_list =  ['an', 'a', 'the']
emojp = [';)', ':)', ':-)', '=)', ':D']
emojn = [':(', ':-(', '=(']
negtag_ = ['not', 'no', 'none', 'neither', 'never', 'nobody']

#--------------Preprocessing l1 & l2
emot_str = r"(?:[:=;][oO\-]?[D\)\]\(\]/\\OpP])"     # Emoticons [eyes][nose][mouth]+[double mouth(( or ))] --> +[\(\(\)\)]
htag_re = r"(?:\#+[\w_]+[\w\'_\-.]*[\w_]+)"         # Hashtags 
#cashtag_re = r'(?:\$+[a-z]*[a-z])'                  # Cashtags '$tikers'
cashtag_re = r'(?:\$[\w]+[\.\_]?[\w]*)'
htmltag_re = r'<[^>]+>'                             # Html tags '<html>'
#htmltag_re = r'(<[^>]+>)+'
mention_re = r'(?:@[\w_]+[\w\'_]*[\w_])'            # Mentions of users '@users'
    
urls_re = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+' #URL (ex: 'http://stks.co/roOMm')
num_re = r'(?:[\-\+\.$\(]*?(?:\d+,?)+(?:\.?\d+\(?)?)'     # Numbers include also price (ex: $45.6) and numbers in ()
owords_re = r'(?:[\w_][\w\'_\-./]*[\w_])+'          # Other words
any_re = r'(?:\S+)'                                 # Anything else

#---Compile token l1 & l2
regex_re = [htag_re, cashtag_re, htmltag_re, mention_re, urls_re, owords_re, any_re]
tokens_re = re.compile(r'('+'|'.join(regex_re)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emot_str+'$', re.VERBOSE | re.IGNORECASE)

#---Compile l1 & l2 
cashtag_re = re.compile(cashtag_re)
num_re = re.compile(num_re)
mention_re = re.compile(mention_re)
urls_re = re.compile(urls_re)


def tokenize_tr(raw_text):
    """
        Data preprocessing for l1 & l2 sentiment analysis 
    """
    # Find all tokens
    tokens = tokens_re.findall(raw_text)
    # Replace emoticons and lowercase tokens
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords_list] 
    # Replace tokens matching num_re with 'numbertag'
    tokens = ['numbertag' if num_re.match(token) else token for token in tokens]
    # Replace tokens matching cashtag_re with 'cashtag'
    tokens = ['cashtag' if cashtag_re.match(token) else token for token in tokens]
    # Replace tokens matching mention_re with 'usertag'
    tokens = ['usertag' if mention_re.match(token) else token for token in tokens]
    # Replace tokens matching urls_re with 'linktag'
    tokens = ['linktag' if urls_re.match(token) else token for token in tokens]
    # Replace tokens in emojn with 'emojineg'
    tokens = ['emojineg' if token in emojn else token for token in tokens]
    # Replace tokens in emojp with 'emojipos'
    tokens = ['emojipos' if token in emojp else token for token in tokens]
    # Replace tokens in negtag_ with 'negtag_'
    tokens = ['negtag_' if token in negtag_ else token for token in tokens]
    # Join all tokens into a string
    tokens = ' '.join(tokens)+' '
    # Attach 'negtag_' to the following word
    tokens = tokens.replace('negtag_ ', 'negtag_')
    # Remove stop words and the preceding space
    tokens = tokens.replace(' an ', ' ')
    tokens = tokens.replace(' a ', ' ')
    tokens = tokens.replace(' the ', ' ')
    return tokens


def get_sent_score(tweet):
    """
        return a sentiment score [-1, 1] for a given tweet
    """
    tokenized_twt = tokenize_tr(tweet)
    bigrams = [b for l in [tokenized_twt] for b in zip(l.split()[:-1], l.split()[1:])]
    bigrams = [" ".join(b) for b in bigrams]
    sent_list = []
    # 1) check for bigrams in the tweets
    for bg in bigrams:
        if bg in df_lexicon.keyword.values:
            sent_list += [df_lexicon.loc[df_lexicon.keyword==bg,"sw"].values[0]]
            # remove bigram from the string to avoid double counting with unigrams
            tokenized_twt = tokenized_twt.replace(bg, "")
    # 2) check unigrams
    unigrams = tokenized_twt.split()
    for ug in unigrams:
        if ug in df_lexicon.keyword.values:
            sent_list += [df_lexicon.loc[df_lexicon.keyword==ug,"sw"].values[0]]
    # return the average value
    try:
        return sum(sent_list) / len(sent_list)
    except ZeroDivisionError:
        return 0

