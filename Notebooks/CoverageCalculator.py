from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import numpy as np

from FeatureExtraction import *


def get_transformed_data(data, pipeline, negative=True, neg_to_pos_ratio=2, stopwords=None):
    features = data.drop('label', axis=1)
    labels = data['label']
    transformed_data = pipeline.named_steps['features'].transform(features)
    
    if stopwords != None:
        indexes = get_feature_index(stopwords, pipeline)   
        for i in indexes:
            transformed_data[:, i] = 0
    
    positive_indices = data[data.label == 1].index.values
    positive_set = transformed_data[positive_indices, :]
    
    if negative == True:
        sample_rate = (float)(len(data[data.label == 1])) / len(data[data.label == 0])
        print("#pos/#neg = {0}".format(sample_rate))

        negative_subset_index = data[data.label == 0].sample(frac=neg_to_pos_ratio * sample_rate, random_state=123).index.values
        
        negative_set = transformed_data[negative_subset_index, :]
        
        return positive_set, negative_set

    return positive_set

def subset_transformed_data(positive_set, n, negative_set=None, neg_to_pos_ratio=2):
    positive_subset = positive_set[np.random.choice(positive_set.shape[0], n, replace=False), :]
    
    if negative_set != None:
        negative_subset = negative_set[np.random.choice(negative_set.shape[0], n * neg_to_pos_ratio, replace=False), :]
        
        return positive_subset, negative_subset
   
    return positive_subset


def get_coverage(pipeline, positive_set, negative_set=None, mutual_info=False):

    negative = (negative_set != None)
    
    num_positive_tweets = positive_set.shape[0]
    if negative == True:
        num_negative_tweets = negative_set.shape[0]
    num_features = positive_set.shape[1]

    feature_positive_coverage = []
    feature_negative_coverage = []
    mutual_info_scores = []
    
    for i in range(num_features):
        feature_positive_coverage.append([])
        if negative == True:
            feature_negative_coverage.append([])
    
    pos_nz = positive_set.nonzero()
    curr_f = 0
    num_nz = len(pos_nz[0])
    i = 0 
    
    for i in range(num_nz):
        f = pos_nz[1][i]
        t = pos_nz[0][i]
        feature_positive_coverage[f].append(t)
        
    if negative == True:
        neg_nz = negative_set.nonzero()
        curr_f = 0
        num_nz = len(neg_nz[0])
        i = 0 

        for i in range(num_nz):
            f = neg_nz[1][i]
            t = neg_nz[0][i]
            feature_negative_coverage[f].append(t)
    
    if mutual_info == True:
        for i in range(num_features):
            mi = normalized_mutual_info_score(labels, transformed_data[:, i].toarray().flatten())
            mutual_info_scores.append(mi)
                 
    if negative == False:
        if mutual_info == False:
            return num_positive_tweets, feature_positive_coverage
        else:
            return num_positive_tweets, feature_positive_coverage, mutual_info_scores
    else:
        if mutual_info == False:
            return num_positive_tweets, feature_positive_coverage, num_negative_tweets, feature_negative_coverage
        else:
            return num_positive_tweets, feature_positive_coverage, num_negative_tweets, feature_negative_coverage, mutual_info_scores