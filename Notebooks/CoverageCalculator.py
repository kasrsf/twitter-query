from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from FeatureExtraction import *

def get_coverage(data, pipeline, negative=False, mutual_info=False, remove_empties=True, positives=None):
    
    features = data.drop('label', axis=1)
    labels = data['label']
    transformed_data = pipeline.named_steps['features'].transform(features)

    if positives == None:
        positive_indices = data.index[data.loc[:, 'label'] == 1]
    else:
        positive_indices = data[data.label == 1].sample(n=positives, random_state=123).index
    positive_set = transformed_data[positive_indices]

    if (negative == True):
        sample_rate = (float)(len(data[data.label == 1])) / len(data[data.label == 0])
        print("#pos/#neg = {0}".format(sample_rate))

        if positives == None:
            negative_subset_index = data[data.label == 0].sample(frac=sample_rate, random_state=123).index
        else:
            negative_subset_index = data[data.label == 0].sample(n=positives, random_state=123).index
            
        negative_set = transformed_data[negative_subset_index]
        num_negative_tweets = negative_set.shape[0]
                 
    num_positive_tweets = positive_set.shape[0]
    num_features = positive_set.shape[1]

    feature_positive_coverage = []
    feature_negative_coverage = []
    mutual_info_scores = []

    for i in range(num_features):
        positive_coverage = positive_set[:, i].nonzero()[0]
        
        if negative == True:
            negative_coverage = negative_set[:, i].nonzero()[0]

        if mutual_info == True:
            if len(positive_coverage) < 5:
                mi = 0
            else:
                mi = normalized_mutual_info_score(labels, transformed_data[:, i].toarray().flatten())

        feature_positive_coverage.append(positive_coverage)
        if negative == True:         
            feature_negative_coverage.append(negative_coverage)
        if mutual_info == True:         
            mutual_info_scores.append(mi)
                 
    # Remove empty_location and empty_mentions from coverages             
    if remove_empties == True:
        empty_loc_index, empty_mention_index = get_empty_feature_indexes(pipeline)
        
        feature_positive_coverage[empty_loc_index] = []
        feature_positive_coverage[empty_mention_index] = []
                 
        if negative == True:
            feature_negative_coverage[empty_loc_index] = []
            feature_negative_coverage[empty_mention_index] = []  
                 
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