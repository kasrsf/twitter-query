from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import numpy as np

from FeatureExtraction import *


def get_transformed_data(data, pipeline, negative=True, neg_to_pos_ratio=2, stopwords=None):
    sample_rate = (float)(len(data[data.label == 1])) / len(data[data.label == 0])
    print("#pos/#neg = {0}".format(sample_rate))

    pos_data = data[data.label == 1]
    neg_data = data[data.label == 0].sample(frac=neg_to_pos_ratio * sample_rate, random_state=123)
    
    pos_features = pos_data.drop('label', axis=1)
    neg_features = neg_data.drop('label', axis=1)
    
    print(len(pos_features))
    print(len(neg_features))
    positive_set = pipeline.named_steps['features'].transform(pos_features)
    negative_set = pipeline.named_steps['features'].transform(neg_features)
    
    print("transform done!")
    if stopwords != None:
        indexes = get_feature_index(stopwords, pipeline)   
        for i in indexes:
            positive_set[:, i] = 0
            negative_set[:, i] = 0
            
    if negative == True:
        return positive_set, negative_set

    return positive_set

def subset_transformed_data(positive_set, n, negative_set=None, neg_to_pos_ratio=2):
    if positive_set.shape[0] < n:
        return positive_set, negative_set
    
    positive_subset = positive_set[np.random.choice(positive_set.shape[0], n, replace=False), :]
    
    if negative_set != None:
        negative_subset = negative_set[np.random.choice(negative_set.shape[0], n * neg_to_pos_ratio, replace=False), :]
        
        return positive_subset, negative_subset
   
    return positive_subset

def get_mi_scores(data, pipeline, pos_coverage):
    features = data.drop('label', axis=1)
    labels = data['label']
    transformed_data = pipeline.named_steps['features'].transform(features)
    
    num_features = transformed_data.shape[1]
    mutual_info_scores = []
    
    for i in range(num_features):
        cov = pos_coverage[i]
        if len(cov) > 50:
            mi = normalized_mutual_info_score(labels, transformed_data[:, i].toarray().flatten())
        else:
            mi = 0
        mutual_info_scores.append(mi)
        
    return mutual_info_scores

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
                
    if negative == False:
            return num_positive_tweets, feature_positive_coverage
        
    return num_positive_tweets, feature_positive_coverage, num_negative_tweets, feature_negative_coverage
