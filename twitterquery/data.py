# fraction=0.1
import numpy as np
import pandas as pd
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, explode, lit, lower
from pyspark.sql import functions as F
from sklearn.metrics import normalized_mutual_info_score

import settings
import transformers
import utils
from hashtag_dict import topic_dict

def get_raw_data(spark_session, directory, fraction=1, 
                 remove_empty_hashtags=True, random_seed=settings.RANDOM_SEED):
    if fraction < 1:
        data = (spark_session
                .read.parquet(directory)
                .sample(False, fraction, random_seed))
    else:
        data = spark_session.read.parquet(directory)

    if remove_empty_hashtags:
        data = data.filter(col('hashtag') != 'empty_hashtag')

    return data

def labeled_data_column_subset(labeled_data):
    return labeled_data.select(#lower(col('from_user')).alias('from_user'),
                               lower(col('hashtag')).alias('hashtags'),
                               lower(col('location')).alias('location'),
                               lower(col('mention')).alias('mention'),
                               #col('tweet_id'),
                               col('term'),
                               col('create_time'),
                               col('label'))

def load_labeled_data(spark_session, data, topic):
    labeled_dir = settings.LABELED_DATA_PARENT_DIR + topic
    pos_dir = labeled_dir + '/pos'
    topical_tweet_ids = spark_session.read.parquet(pos_dir)
    topical_tweets = (topical_tweet_ids
                        .join(data, 'tweet_id')
                        .withColumn('label', lit(1)))
    
    neg_dir = labeled_dir + '/neg'
    non_topical_tweet_ids = (spark_session
                                .read.parquet(neg_dir) 
                                .sample(withReplacement=False,
                                    fraction=0.1,
                                    seed=settings.RANDOM_SEED))                                
    non_topical_tweets = (non_topical_tweet_ids
                            .join(data, 'tweet_id') 
                            .withColumn('label', lit(0)))

    labeled_data = topical_tweets.union(non_topical_tweets)
                                 
    return labeled_data#)

def get_labeled_data(data, topic):
    # Tokenizer converts input to lowercase and then splits by white space
    hashtag_tokenizer = Tokenizer(inputCol='hashtag', outputCol='hashtags')
    hashtag_occurances = (hashtag_tokenizer
                                .transform(data) 
                                .select('tweet_id', 'hashtags')
                                .withColumn('hashtags', explode('hashtags')))

    topic_labels = topic_dict[topic]
    topical_ids = (hashtag_occurances
                    .select(hashtag_occurances.tweet_id.alias('topical_id'))
                    .where(hashtag_occurances.hashtags.isin(topic_labels))
                    .distinct() 
                    .cache())
    
    labeled_topical = topical_ids.withColumn('topical', lit(1))   
    labeled_data = (data
                    .join(other=labeled_topical,
                            on=data.tweet_id == labeled_topical['topical_id'], 
                            how="left")
                    .withColumn('label', F.when(labeled_topical.topical == 1, 1).otherwise(0)))
    
    return labeled_data_column_subset(labeled_data)

def get_num_of_positive_labels(labeled_data):
    return labeled_data.groupBy().sum('label').collect()[0][0]

def split_kfold(labeled_data, k=5, random_seed=settings.RANDOM_SEED):
    split_size = [1.0 / k] * k
    return labeled_data.randomSplit(split_size, seed=random_seed)

def load_splitted_data(topic, 
                        num_of_splits=10,
                        stored_splits_dir=settings.SPLITTED_DATA_PARENT_DIR,
                        shuffle=False,
                        random_seed=settings.RANDOM_SEED):
    merged_data = None
    for i in range(num_of_splits):
        data_path = stored_splits_dir + topic + "/" + str(i+1) + ".csv"
        splitted_df = pd.read_csv(data_path).dropna()
        if merged_data is None:
            merged_data = splitted_df
        else:            
            merged_data = pd.concat([merged_data, splitted_df])
    if shuffle is True:
        merged_data = merged_data.sample(frac=1, random_state=random_seed) \
                        .reset_index()

    return merged_data

def merge_splits_into_train_test(data_splits, test_split_index):
    num_of_splits = len(data_splits)
    train = None
    for i in range(num_of_splits):
        if i == test_split_index:
            test = data_splits[i]
        else:
            if train is None:
                train = data_splits[i]
            else:
                train = pd.concat([train, data_splits[i]])
    return train, test

def get_transformed_data(topic, cached_labeled=None, shuffle=True):
    if cached_labeled is None:
        labeled_data = load_splitted_data(topic, shuffle=shuffle)
    else:
        labeled_data = cached_labeled#.iloc[:10000]
    labeled_data_features = labeled_data.drop('label', axis=1)
    label_data = labeled_data.label
    transformer = transformers.DataframeTransformer(labeling_tags=utils.get_labeling_tags_for_topic(topic))
    transformed_data = transformer.fit_transform(labeled_data_features)
    return (transformed_data, label_data, transformer)

def filter_matrix_by_index(feature_matrix, labels, filtering_indices):
    filtered_rows = set()
    for feature in filtering_indices:
        filtered_rows = filtered_rows.union(set(feature_matrix[:, feature].nonzero()[0]))
    filtered_rows = list(filtered_rows)
    return feature_matrix[filtered_rows, :], labels.loc[filtered_rows]

def get_mi_scores(data_features, data_labels):
    sampled_data, sampled_labels = utils.sample_sparse_matrix(data_features, data_labels)
    num_features = sampled_data.shape[1]
    mutual_info_scores = []
    for i in range(num_features):
        cov = sampled_data[:, i].toarray().flatten()
        feat_mi_score = normalized_mutual_info_score(sampled_labels, cov)
        mutual_info_scores.append(feat_mi_score)
    return mutual_info_scores