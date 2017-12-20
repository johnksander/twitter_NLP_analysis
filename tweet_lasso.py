import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats.mstats import zscore
from datetime import datetime
import os
import csv
import operator
import pickle
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=__doc__
)
parser.add_argument(
    '--dataset',
    help='Specify "full" or "small" Sentiment140 dataset, "small" is example sample.'
)
parser.add_argument(
    '--Fth',
    help='Feature occurance threshold: features observed < Fth in the dataset are removed.',
    default=0
)
parser.add_argument(
    '--workers',
    help='number of workers for multithreaded lasso fitting',
    default=1
)
parser.add_argument(
    '-v',
    '--verbose',
    action='store_true',
    help='verbose lasso fitting, useful for progress tracking longer analyses',
    default=0
)


args = parser.parse_args()
Fth = int(args.Fth)
num_workers = int(args.workers)
data2use = args.dataset
Vopt = 0
if args.verbose:
    Vopt = 1

    
data_dir = 'data' #set up the datafile paths    
if data2use == 'small':
    data_dir = os.path.join(data_dir,'small')
    data_fn = 'word_as_vec.csv'
    info_fn = 'labels.csv'
elif data2use == 'full':
    data_fn = 'twitter_data.csv'
    info_fn = 'token_labels.csv'

def feature_threshold(x):
    """returns boolean for features above occurance threshold"""
    Fcount = np.sum(x,axis=0)
    Fvalid = Fcount > Fth
    return Fvalid    
    
def obs_with_data(x):
    """returns boolean for observations with feature data"""
    num_toks = np.sum(x,axis=1)
    has_data = num_toks > 0
    return has_data
    
def freq_score(x):
    """converts token count features to tweet-frequency features"""
    num_toks = np.sum(x,axis=1)
    return x / num_toks[:,None]

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

def timestamp():
    """gives a formatted timestamp, useful for longer analyses"""
    return datetime.now().strftime('%H:%M:%S %m-%d')

def save_mdl(x):
    """saves a pickled model file"""
    f = open('trained_lasso_mdl.pkl', 'wb')
    pickle.dump(x, f, -1)
    f.close()
    
def show_betas(B,ID):
    """Prints sorted beta-token pairs to a txt file, then prints a figure
    of the beta coefficient distribution"""
    beta_dict = dict(zip(ID, B))
    f = open('feature_betas.txt','w')
    for k in sorted(beta_dict, key=lambda x: beta_dict[x]):
        f.write("B {} = {}\n".format(k, beta_dict[k]))
    f.close()
    #show beta distribution 
    fig = plt.figure(dpi=300)
    plt.hist(betas,bins=80)
    plt.xlabel('Beta')
    plt.ylabel('features')
    plt.savefig('betas.png')
    
    

print('Start {}\n'.format(timestamp()))
print('loading {} dataset...'.format(data2use))

tweetset = load_csv(os.path.join(data_dir,data_fn))
print('dataset loaded {}\n'.format(timestamp()))
print('N tweets = {}'.format(tweetset.shape[0]))
print('N features = {}'.format(tweetset.shape[1]-1))

with open(os.path.join(data_dir,info_fn), 'r') as f:
    featIDs = list(csv.reader(f))    
featIDs = featIDs[0]
featIDs = featIDs[1:] #get rid of "LABEL" header

valid_feats = feature_threshold(tweetset[:,1:])
#add in a True boolean to match with "LABEL" header spot 
valid_feats = np.concatenate((np.array([True]),valid_feats))
tweetset = tweetset[:,valid_feats] #remove features < Fth
print('------N features < {} count = {}'.format(Fth,(len(valid_feats)-1) - (tweetset.shape[1]-1)))
print('---N valid features for analysis = {}\n'.format(tweetset.shape[1]-1))

valid_obs = obs_with_data(tweetset[:,1:])
tweetset = tweetset[valid_obs,:] #remove obs without feature data
print('---N tweets without feature data = {}'.format(len(valid_obs) - tweetset.shape[0]))
print('---N valid tweets for analysis = {}\n'.format(tweetset.shape[0]))

#seperate labels from feature data
labels = tweetset[:,0]
Nobs = labels.shape[0]
data_matrix = tweetset[:,1:]
Nfeats = data_matrix.shape[1]

data_matrix = freq_score(data_matrix) #convert features to tweet-frequency scores
data_matrix = zscore(data_matrix) #standardize predictors

#find best lambda regularization with 10 fold CV 
mdl = LogisticRegressionCV(n_jobs=num_workers, penalty='l1', solver='liblinear', cv=10,                           scoring='accuracy', random_state=0, verbose=Vopt)
mdl.fit(data_matrix, labels) #refit model on optimized parameters 

print("\nCV accruacy = {:.2%}\n".format(mdl.score(data_matrix, labels)))

save_mdl(mdl)

betas = mdl.coef_[0]
num_dropped = len(list(filter(None,betas == 0)))
print("---N features = {}\n---N B0 = {}".format(Nfeats,num_dropped))

show_betas(betas,featIDs)

