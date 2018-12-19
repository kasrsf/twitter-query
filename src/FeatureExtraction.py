import numpy as np
import pandas as pd

def get_pipeline_features(pipeline):
    term_transformer = pipeline.named_steps['features'].transformer_list[0][1]
    term_features = term_transformer.named_steps['count'].get_feature_names()
    
    hashtag_transformer = pipeline.named_steps['features'].transformer_list[1][1]
    hashtag_features = hashtag_transformer.named_steps['count'].get_feature_names()
    
    user_transformer = pipeline.named_steps['features'].transformer_list[2][1]
    user_features = user_transformer.named_steps['count'].get_feature_names()
    
    location_transformer = pipeline.named_steps['features'].transformer_list[3][1]
    location_features = location_transformer.named_steps['count'].get_feature_names()
    
    mention_transformer = pipeline.named_steps['features'].transformer_list[4][1]
    mention_features = mention_transformer.named_steps['count'].get_feature_names()
    
    return term_features, hashtag_features, user_features, location_features, mention_features

def get_feature_index(features, pipeline):
    term_features, hashtag_features, user_features, location_features, mention_features = \
                       get_pipeline_features(pipeline)   
        
    all_features = [term_features, hashtag_features, user_features, location_features, mention_features]
    all_features = [feature for features_list in all_features for feature in features_list]        
    
    indexes = []
    for feature in features:
        if feature in all_features:
            indexes.append(all_features.index(feature))
       
    return indexes   

def get_empty_feature_indexes(pipeline):
    term_features, hashtag_features, user_features, location_features, mention_features = \
                        get_pipeline_features(pipeline)   
        
    prev_length = len(term_features) + len(hashtag_features) + len(user_features)
    empty_location_index = prev_length + location_features.index("empty_location")
    empty_mention_index = prev_length + len(location_features) + mention_features.index("empty_mention")
    
    return empty_location_index, empty_mention_index

def get_feature_by_index(pipeline, indexes, pos_coverages=None, neg_coverages=None, mutual_info_scores=None):
    term_features, hashtag_features, user_features, location_features, mention_features = \
                        get_pipeline_features(pipeline)
    
    term_last_index = len(term_features)
    hash_last_index = term_last_index + len(hashtag_features)
    user_last_index = hash_last_index + len(user_features)
    loc_last_index = user_last_index + len(location_features)
    
    features_df = pd.DataFrame()
    for i in range(len(indexes)):
        if indexes[i] < term_last_index:
            feature = term_features[indexes[i]]
            features_df = features_df.append([[feature, "Term"]])
        elif indexes[i] < hash_last_index:
            feature = hashtag_features[indexes[i] - term_last_index]
            features_df = features_df.append([[feature, "Hashtag"]])
        elif indexes[i] < user_last_index:
            feature = user_features[indexes[i] - hash_last_index]
            features_df = features_df.append([[feature, "User"]])
        elif indexes[i] < loc_last_index:
            feature = location_features[indexes[i] - user_last_index]
            features_df = features_df.append([[feature, "Location"]])
        else:
            feature = mention_features[indexes[i] - loc_last_index]
            features_df = features_df.append([[feature, "Mention"]])
    
    features_df = features_df.reset_index(drop=True)
    
    
    if pos_coverages == None:
        features_df.columns = ["Feature", "Type"]
    else:
        selected_coverages = [len(pos_coverages[i]) for i in indexes]
        
        features_df = pd.concat([features_df, pd.DataFrame(selected_coverages)], axis=1)
        features_df.columns = ["Feature", "Type", "Positive Coverage"]
        
    if neg_coverages != None:
        selected_coverages = [len(neg_coverages[i]) for i in indexes]
        
        features_df = pd.concat([features_df, pd.DataFrame(selected_coverages)], axis=1)
        features_df = features_df.rename(columns = {0: "Negative Coverage"})
        
    if mutual_info_scores != None:
        selected_mis = [mutual_info_scores[i] for i in indexes]
        
        features_df = pd.concat([features_df, pd.DataFrame(selected_mis)], axis=1)
        features_df = features_df.rename(columns = {0: "MI Score"})
        
    return features_df

def topk_features(pipeline, k=50):
    clf = pipeline.named_steps['classifier']
    term_features, hashtag_features, user_features, location_features, mention_features \
                                                                                = get_pipeline_features(pipeline)
    
    feature_names = term_features + hashtag_features + user_features + location_features + mention_features

    topk = np.flipud(np.argsort(clf.coef_[0])[-k:])
    return topk
    
# Currently only get top terms+hashtags and users Ordering= ['term+hashtag', 'user', 'location', 'mention']
def top_features_by_kind(pipeline, k=[50, 50, 50, 0, 0]):
    clf = pipeline.named_steps['classifier']
    term_features, hashtag_features, user_features, location_features, mention_features \
                                                                                = get_pipeline_features(pipeline)
    if k[0] > 0:
        last_term_index = len(term_features) 
        term_coeffs = clf.coef_[0][:last_term_index]
        top_terms_index = np.flipud(np.argsort(term_coeffs)[-k[0]:])
        top_terms = [term_features[j] for j in top_terms_index]
    else:
        last_term_index = 0
        top_terms = None

    if k[1] > 0:
        last_hashtag_index = last_term_index + len(hashtag_features)
        hashtag_coeffs = clf.coef_[0][last_term_index:last_hashtag_index]
        top_hashtags_index = np.flipud(np.argsort(hashtag_coeffs)[-k[1]:])
        top_hashtags = [hashtag_features[j] for j in top_hashtags_index]  
    else:
        last_hashtag_index = max(0, last_term_index)
        top_hashtags = None
    
    if k[2] > 0:
        last_user_index = last_hashtag_index + len(user_features)
        user_coeffs = clf.coef_[0][last_hashtag_index:last_user_index]
        top_users_index = np.flipud(np.argsort(user_coeffs)[-k[2]:])
        top_users = [user_features[j] for j in top_users_index]
    else:
        last_user_index = max(0, last_hashtag_index)
        top_users = None
    
    if k[3] > 0:
        last_location_index = last_user_index + len(location_features)
        location_coeffs = clf.coef_[0][last_user_index:last_location_index]
        top_locs_index = np.flipud(np.argsort(location_coeffs)[-k[3]:])
        top_locs = [location_features[j] for j in top_locs_index]
    else:
        last_location_index = max(0, last_user_index)        
        top_locs = None        
    
    if k[4] > 0:
        last_mention_index = last_location_index + len(mention_features)
        mention_coeffs = clf.coef_[0][last_location_index:last_mention_index]
        top_mentions_index = np.flipud(np.argsort(mention_coeffs)[-k[4]:])
        top_mentions = [mention_features[j] for j in top_mentions_index]
    else:
        last_hashtag_index = max(0, last_location_index)        
        top_mentions = None       

    return top_terms, top_hashtags, top_users, top_locs, top_mentions