import re
import sys
import csv
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import porter

STOPWORDS = set(stopwords.words('english'))
STEMMER = porter.PorterStemmer()
LIMIT = 500


def clean(tweet):
    """ Return a list of words """

    # clean hashtags, twitter names, web addresses, puncuation
    tweet = (re.sub(r"#[\w\d]*|@[.]?[\w\d]*[\'\w*]*|https?:\/\/\S+\b|\
             www\.(\w+\.)+\S*|[.,:;!?()$-/^]*", "", tweet).lower())

    # strip repeated chars (extra vals)
    tweet = re.sub(r"(.)\1\1{1,1}", "", tweet)
    tweet = (re.sub(r"($.)\1{1,}", "", tweet).split())

    tweet = [STEMMER.stem_word(x) for x in tweet if
             x not in STOPWORDS and len(x) > 1]
    return tweet


if __name__ == "__main__":
    """ Builds a word vector based on tweets.  File takes 4 arguments
       file_in file_out, label_column, tweet_column
    """

    file = open(sys.argv[1], encoding="ISO-8859-1")

    label_row = int(sys.argv[3])
    tweet_row = int(sys.argv[4])

    pos = []
    neg = []
    tweets = []

    cv = csv.reader(file)

    for line in cv:

        tweet = clean(line[tweet_row])

        if len(tweet) > 0:
            tweets.append((tweet, line[label_row]))

        else:
            continue

        if line[label_row] == '0':
            neg.extend(tweet)
        else:
            pos.extend(tweet)

    good_count = Counter(pos)
    bad_count = Counter(neg)

    # filter words by limit
    neg_words = [x for x, y in bad_count.items() if y > LIMIT]
    pos_words = [x for x, y in good_count.items() if y > LIMIT]

    # combine negative and positive words
    tot = neg_words + pos_words
    tot = list(set(tot))
    tot.sort()

    outfile = open(sys.argv[2], 'w')
    headers = 'LABEL' + ',"' + '","'.join(tot) + '"\n'
    outfile.write(headers)

    # total and count for printing progress
    total = len(tweets)
    count = 0.0

    for tweet, label in tweets:

        count += 1
        if count % 100000 == 0:
            print (count, "(", count/total, '% done )')

        tweet_matrix = [label]
        tweet_matrix += [0 for x in tot]
        for word in tweet:
            if word in tot:
                tweet_matrix[tot.index(word)+1] += 1

        str_out = ','.join(map(str, tweet_matrix)) + '\n'
        outfile.write(str_out)
