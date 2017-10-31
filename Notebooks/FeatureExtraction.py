import numpy as np

def get_pipeline_features(pipeline):
    term_transformer = pipeline.named_steps['features'].transformer_list[0][1]
    term_features = term_transformer.named_steps['tfidf'].get_feature_names()
    
    hashtag_transformer = pipeline.named_steps['features'].transformer_list[1][1]
    hashtag_features = hashtag_transformer.named_steps['count'].get_feature_names()
    
    user_transformer = pipeline.named_steps['features'].transformer_list[2][1]
    user_features = user_transformer.named_steps['count'].get_feature_names()
    
    location_transformer = pipeline.named_steps['features'].transformer_list[3][1]
    location_features = location_transformer.named_steps['count'].get_feature_names()
    
    mention_transformer = pipeline.named_steps['features'].transformer_list[4][1]
    mention_features = mention_transformer.named_steps['count'].get_feature_names()
    
    return term_features, hashtag_features, user_features, location_features, mention_features

def topk_features(pipeline, k=50):
    clf = pipeline.named_steps['classifier']
    term_features, hashtag_features, user_features, location_features, mention_features \
                                                                                = get_pipeline_features(pipeline)
    
    feature_names = term_features + hashtag_features + user_features + location_features + mention_features

    topk = np.flipud(np.argsort(clf.coef_[0])[-k:])
    return [feature_names[j] for j in topk]
    
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
        top_terms = None

    if k[1] > 0:
        last_hashtag_index = last_term_index + len(hashtag_features)
        hashtag_coeffs = clf.coef_[0][last_term_index:last_hashtag_index]
        top_hashtags_index = np.flipud(np.argsort(hashtag_coeffs)[-k[1]:])
        top_hashtags = [hashtag_features[j] for j in top_hashtags_index]  
    else:
        top_hashtags = None
    
    if k[2] > 0:
        last_user_index = last_hashtag_index + len(user_features)
        user_coeffs = clf.coef_[0][last_hashtag_index:last_user_index]
        top_users_index = np.flipud(np.argsort(user_coeffs)[-k[2]:])
        top_users = [user_features[j] for j in top_users_index]
    else:
        top_users = None
    
    if k[3] > 0:
        last_location_index = last_user_index + len(location_features)
        location_coeffs = clf.coef_[0][last_user_index:last_location_index]
        top_locs_index = np.flipud(np.argsort(location_coeffs)[-k[3]:])
        top_locs = [location_features[j] for j in top_locs_index]
    else:
        top_locs = None        
    
    if k[4] > 0:
        last_mention_index = last_location_index + len(mention_features)
        mention_coeffs = clf.coef_[0][last_location_index:last_mention_index]
        top_mentions_index = np.flipud(np.argsort(mention_coeffs)[-k[4]:])
        top_mentions = [mention_features[j] for j in top_mentions_index]
    else:
        top_mentions = None       

    return top_terms, top_hashtags, top_users, top_locs, top_mentions