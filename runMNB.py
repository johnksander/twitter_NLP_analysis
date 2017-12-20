import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
import os
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=__doc__
)
parser.add_argument(
    '--dataset',
    help='Specify "full" or "small" Sentiment140 dataset, "small" is example sample.'
)

args = parser.parse_args()
data2use = args.dataset
    
data_dir = 'data'
if data2use == 'small':
    data_dir = os.path.join(data_dir,'small')
    data_fn = 'word_as_vec.csv'
    info_fn = 'labels.csv'
elif data2use == 'full':
    data_fn = 'twitter_data.csv'
    info_fn = 'token_labels.csv'


def obs_with_data(x):
    """returns boolean for observations with feature data"""
    num_toks = np.sum(x,axis=1)
    has_data = num_toks > 0
    return has_data

def timestamp():
    """gives a formatted timestamp, useful for longer analyses"""
    return datetime.now().strftime('%H:%M:%S %m-%d')

def load_csv(fn):
    """Iteratively loads csv file to a numpy array. This is needed for
    larger datafiles."""
    def iter_func():
        with open(fn, 'r') as infile:
            for line in infile:
                line = line.rstrip().split(',')
                for item in line:
                    yield float(item)
        load_csv.rowlength = len(line)
    data = np.fromiter(iter_func(), dtype=float)
    data = data.reshape((-1, load_csv.rowlength))
    return data

            
print('loading {} dataset...'.format(data2use))
tweetset = load_csv(os.path.join(data_dir,data_fn))
print('dataset loaded\n')
print('N tweets = {}'.format(tweetset.shape[0]))
print('N features = {}'.format(tweetset.shape[1]))

valid_obs = obs_with_data(tweetset[:,1:])
tweetset = tweetset[valid_obs,:] #remove obs without feature data

print('---N tweets without feature data = {}'.format(len(valid_obs) - tweetset.shape[0]))
print('---N valid tweets for analysis = {}\n'.format(tweetset.shape[0]))

#seperate labels from feature data
labels = tweetset[:,0]
Nobs = labels.shape[0]
data_matrix = tweetset[:,1:]

#set up 10 fold cross-validation 
k_folds = 10
CVinds = StratifiedKFold(n_splits=k_folds)
print('setting up k={} fold cross-validation...'.format(k_folds))
curr_fold = 0 #for progress tracking

CVinds.split(data_matrix,labels)
guesses = np.empty(labels.shape)
true_labels = np.empty(labels.shape)
for train_inds, test_inds in CVinds.split(data_matrix,labels):
    MNB = MultinomialNB()
    MNB.fit(data_matrix[train_inds,:],labels[train_inds])
    guesses[test_inds] = MNB.predict(data_matrix[test_inds,:])
    true_labels[test_inds] = labels[test_inds]
    curr_fold = curr_fold + 1 
    print('---fold #{}/{} complete    {}'.format(curr_fold,k_folds,timestamp()))


num_correct = len(list(filter(None,guesses == true_labels)))
accuracy = num_correct / Nobs
print("\nCV accruacy = {:.2%}\n".format(accuracy))


