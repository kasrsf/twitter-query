import numpy as np
import pandas as pd
import random
from scipy.sparse import vstack
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

import settings
import utils

class ItemSelector(TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self
    
    def get_feature_names(self):
        return self.key

    def transform(self, data_dict):
        return data_dict[self.key].values.astype('str')

class DataframeTransformer:
    def __init__(self, labeling_tags=None):
        self.labeling_tags = labeling_tags
        self.feature_transformer = FeatureUnion(
            transformer_list=[
                ('term', Pipeline([
                    ('selector', ItemSelector(key='term')),
                    ('count', CountVectorizer())])),
                ('hashtag', Pipeline([
                    ('selector', ItemSelector(key='hashtags')),
                    ('count', CountVectorizer(stop_words=self.labeling_tags, ))])),
                ('user', Pipeline([
                    ('selector', ItemSelector(key='from_user')),
                    ('count', CountVectorizer())])),
                ('location', Pipeline([
                    ('selector', ItemSelector(key='location')),
                    ('count', CountVectorizer(stop_words=['empty_location'], ))])),
                ('mention', Pipeline([
                    ('selector', ItemSelector(key='mention')),
                    ('count', CountVectorizer(stop_words=['empty_mention'], ))]))])

    def fit(self, data):        
        self.feature_transformer.fit(data)
    
    def fit_transform(self, data):
        return self.feature_transformer.fit_transform(data)

    def transform(self, data):
        return self.feature_transformer.transform(data)

    def get_pipeline_features(self):
        term_transformer = self.feature_transformer.transformer_list[0][1]
        terms = term_transformer.named_steps['count'].get_feature_names()
        terms_df = utils.get_features_pd_from_list(terms, 'term')
    
        hashtag_transformer = self.feature_transformer.transformer_list[1][1]
        hashtags = hashtag_transformer.named_steps['count'].get_feature_names()
        hashtags_df = utils.get_features_pd_from_list(hashtags, 'hashtag')
        
        user_transformer = self.feature_transformer.transformer_list[2][1]
        users = user_transformer.named_steps['count'].get_feature_names()
        users_df = utils.get_features_pd_from_list(users, 'user')
        
        location_transformer = self.feature_transformer.transformer_list[3][1]
        locations = location_transformer.named_steps['count'].get_feature_names()
        locations_df = utils.get_features_pd_from_list(locations, 'location')
        
        mention_transformer = self.feature_transformer.transformer_list[4][1]
        mentions = mention_transformer.named_steps['count'].get_feature_names()
        mentions_df = utils.get_features_pd_from_list(mentions, 'mention')

        features_df = pd.concat([terms_df, hashtags_df, users_df, locations_df, mentions_df]) \
                        .reset_index(drop=True)
        return features_df

    def get_features_by_index(self, indices):
        features_df = self.get_pipeline_features()
        return features_df.iloc[indices, :].reset_index(drop=True)

def train_test_split(occurance_matrix,
                     labels,
                     test_split_index,
                     num_of_splits=5):
    split_size = occurance_matrix.shape[0] / num_of_splits
    train_features = None
    for i in range(num_of_splits):
        split_data = occurance_matrix[i * split_size:(i + 1) * split_size]
        split_labels = labels[i * split_size:(i + 1) * split_size]
        if i == test_split_index:
            test_features = split_data
            test_labels = split_labels.reset_index(drop=True)
        else:
            if train_features is None:
                train_features = split_data
                train_labels = split_labels
            else:
                train_features = vstack([train_features, split_data])
                train_labels = train_labels.append(split_labels)
    return (train_features, train_labels.reset_index(drop=True),
             test_features, test_labels)

def get_positive_negative_set(occurance_matrix, labels, pos_count=None, neg_to_pos_ratio=1):
    random.seed(settings.RANDOM_SEED)
    pos_idx = list(labels.nonzero()[0])
    if pos_count != None and len(pos_idx) > pos_count:
        pos_idx = random.sample(pos_idx, pos_count)
    neg_idx = list(set(range(len(labels))) - set(pos_idx))
    if neg_to_pos_ratio > 0:
        num_of_negatives = len(pos_idx) * neg_to_pos_ratio
        neg_idx = random.sample(neg_idx, num_of_negatives)
    positive_set = occurance_matrix[pos_idx, :]
    negative_set = occurance_matrix[neg_idx, :]
    return positive_set, negative_set