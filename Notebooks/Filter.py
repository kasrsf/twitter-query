import pandas as pd
import re


def filter_data(data, terms=[], hashtags=None, users=None, locs=None, mentions=None):
    filtered_data = pd.DataFrame()

    if len(terms) > 0:
        terms_filter = [re.sub(r'(.*)', r'\\b\1\\b', term) for term in terms]
        terms_regex = '|'.join(terms_filter)
        filtered_data = data.loc[data['term'].str.lower().str.contains(terms_regex)]

    if hashtags != None and len(hashtags) > 0:
        hashtags_filter = [re.sub(r'(.*)', r'\\b\1\\b', hashtag) for hashtag in hashtags]
        hashtags_regex = '|'.join(hashtags_filter)
        hashtags_filtered = data.loc[data['hashtag'].str.lower().str.contains(hashtags_regex)]
        filtered_data = pd.concat([filtered_data, hashtags_filtered])

    if users != None and len(users) > 0:
        users_filter = [re.sub(r'(.*)', r'\\b\1\\b', user) for user in users]
        users_regex = '|'.join(users_filter)
        users_filtered = data.loc[data['from_user'].str.lower().str.contains(users_regex)]
        filtered_data = pd.concat([filtered_data, users_filtered])

    if locs != None and len(locs) > 0:
        locs_filter = [re.sub(r'(.*)', r'\\b\1\\b', loc) for loc in locs]
        locs_regex = '|'.join(locs_filter)
        locs_filtered = data.loc[data['location'].str.lower().str.contains(locs_regex)]
        filtered_data = pd.concat([filtered_data, locs_filtered])
        
    if mentions != None and len(mentions) > 0:
        mentions_filter = [re.sub(r'(.*)', r'\\b\1\\b', mention) for mention in mentions]
        mentions_regex = '|'.join(mentions_filter)
        mentions_filtered = data.loc[data['mention'].str.lower().str.contains(mentions_regex)]
        filtered_data = pd.concat([filtered_data, mentions_filtered])

    return filtered_data.drop_duplicates(subset='tweet_id')