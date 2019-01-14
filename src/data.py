# fraction=0.1
import numpy as np
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import col, explode, lit, lower
from pyspark.sql import functions as F

from hashtag_dict import topic_dict

LABELED_DATA_PARENT_DIR = '/mnt/1e69d2b1-91a9-473c-a164-db90daf43a3d/labeled_data/'
RANDOM_SEED = 42

def get_raw_data(spark_session, directory, fraction=1, 
                 remove_empty_hashtags=True, random_seed=123):
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
    return labeled_data.select(lower(col('from_user')).alias('from_user'),
                               lower(col('hashtag')).alias('hashtags'),
                               lower(col('location')).alias('location'),
                               lower(col('mention')).alias('mention'),
                               col('tweet_id'),
                               col('term'),
                               col('label'))

def load_labeled_data(spark_session, data, topic):
    labeled_dir = LABELED_DATA_PARENT_DIR + topic
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
                                    seed=RANDOM_SEED))                                
    non_topical_tweets = (non_topical_tweet_ids
                            .join(data, 'tweet_id') 
                            .withColumn('label', lit(0)))

    labeled_data = topical_tweets.union(non_topical_tweets)
                                 
    return labeled_data_column_subset(labeled_data)

def get_labeled_data(data, topic, load_labeled=False):
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

def split_kfold(labeled_data, k=5, random_seed=RANDOM_SEED):
    split_size = [1.0 / k] * k
    print(split_size)
    return labeled_data.randomSplit(split_size, seed=random_seed)

def get_num_of_positive_labels(labeled_data):
    return labeled_data.groupBy().sum('label').collect()[0][0]
    