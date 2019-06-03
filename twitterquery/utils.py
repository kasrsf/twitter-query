import pandas as pd
import random
import scipy
import time
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import scipy.stats as st

import classification
import data
import settings
import utils

def get_features_pd_from_list(features, feature_type):
    df = pd.DataFrame(features)
    df['type'] = feature_type
    df.columns = ['Feature', 'Type']
    return df

def get_labeling_tags():
    return settings.LABELING_TAGS

def get_labeling_tags_for_topic(topic):
    return list(settings.LABELING_TAGS[topic])

def get_topics():
    return settings.TOPICS

def get_readable_topic(topic):
    return settings.TOPICS_HUMAN_READABLE[topic] if topic in settings.TOPICS_HUMAN_READABLE else topic

def get_stat_methods():
    return settings.STAT_METHODS

def get_feature_methods():
    return settings.FEATURE_METHODS

def get_labeled_data_statistics(labels):
    return labels.sum(), len(labels)

def delete_row_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - len(i), mat._shape[1])

def sample_sparse_matrix(matrix, labels=None, n=5000):
    if labels is None:
        all_idx = range(matrix.shape[0])
    else:
        all_idx = labels.index
    sampled_idx = random.sample(all_idx, n)
    sampled_data = matrix[sampled_idx, :]
    if labels is None:
        sampled_labels = None
    else:
        sampled_labels = labels[sampled_idx]
    return sampled_data, sampled_labels
    
def save_results_to_csv(topic, method, phase1, train_avep, test_avep, patk, features=None):
    phase1_df = pd.DataFrame(phase1, columns=['Relevant', 'Total'])
    phase1_df.to_csv(settings.RESULTS_DIR + topic + '/' + method + '/phase1.csv', index=False)

    phase2_df = pd.DataFrame([train_avep, test_avep, patk]).transpose()
    phase2_df.columns = ['Train AveP', 'Test AveP', 'P@100']
    phase2_df.to_csv(settings.RESULTS_DIR + topic + '/' + method + '/phase2.csv', index=False)

    if features is not None:
        for i, feats in enumerate(features):
            feats.to_csv(settings.RESULTS_DIR + topic + '/' + method 
                            + '/feats/{}.csv'.format(i+1), index=False)

def load_results_stats(topic, method):
    # phase1
    # get total positives from firehose
    df_path = str(settings.RESULTS_DIR + topic + '/firehose/phase1.csv')
    total_pos = pd.read_csv(df_path).iloc[:, 0]
    df_path = str(settings.RESULTS_DIR + topic + '/' + method + '/phase1.csv')
    phase1_stats_df = pd.read_csv(df_path)
    phase1_stats_df['Total Relevant'] = total_pos
    phase1_stats_df['Precision'] = phase1_stats_df.iloc[:, 0] / phase1_stats_df.iloc[:, 1]
    phase1_stats_df['Recall'] = phase1_stats_df.iloc[:, 0] / phase1_stats_df.loc[:, 'Total Relevant']
    phase1_stats_df['F1 Score'] = ((2 * phase1_stats_df.loc[:, 'Precision'] * phase1_stats_df.loc[:, 'Recall'])  
                              / (phase1_stats_df.loc[:, 'Precision'] + phase1_stats_df.loc[:, 'Recall']))
    #phase1
    # phase2
    df_path = str(settings.RESULTS_DIR + topic + '/' + method + '/phase2.csv')
    phase2_stats_df = pd.read_csv(df_path)
    return phase1_stats_df, phase2_stats_df

def load_results_features(topic, method, k=5):
    feature_dfs = []
    for i in range(k):
        df_path = str(settings.RESULTS_DIR + topic + '/' + method + '/feats/{}.csv').format(i + 1)
        feature_df = pd.read_csv(df_path)
        feature_dfs.append(feature_df)
    return feature_dfs

def get_confidence_interval_vals(data):
    interval = st.t.interval(0.95, len(data) -1, loc=np.mean(data), scale=st.sem(data))
    if np.isnan(interval[0]):
        interval_range = 0
    else:
        interval_range = (interval[1] - interval[0]) / 2
    return np.mean(data), interval_range

def get_confidence_interval(data):
    interval = st.t.interval(0.95, len(data) -1, loc=np.mean(data), scale=st.sem(data))
    if np.isnan(interval[0]):
        return "{:.3f}".format(np.mean(data))
    else:
        interval_range = (interval[1] - interval[0]) / 2
        return "{:.3f} +\- {:.3f}".format(np.mean(data), interval_range)

def get_confidence_interval_no_decimals(data):
    interval = st.t.interval(0.95, len(data) -1, loc=np.mean(data), scale=st.sem(data))
    if np.isnan(interval[0]):
        return "{:,.0f}".format(np.mean(data))
    else:
        interval_range = (interval[1] - interval[0]) / 2
        return "{:,.0f} +\- {:,.0f}".format(np.mean(data), interval_range)

def summarise_stats(topic, method):
    ph1, ph2 = load_results_stats(topic, method)    
    ph1_no_dec_summarized = ph1.iloc[:, :3].apply(get_confidence_interval_no_decimals, axis=0)
    ph1_dec_summarized = ph1.iloc[:, 3:].apply(get_confidence_interval, axis=0) 
    ph1_summarized = pd.DataFrame(pd.concat([ph1_no_dec_summarized, ph1_dec_summarized])).transpose()
    ph1_summarized.index = [method]
    ph2_summarized = pd.DataFrame(ph2.apply(get_confidence_interval, axis=0)).transpose()
    ph2_summarized.index = [method]

    return ph1_summarized, ph2_summarized

def get_all_stats(topic, methods):
    ph1_stats = pd.DataFrame()
    ph2_stats = pd.DataFrame()
    for method in methods:
        ph1, ph2 = summarise_stats(topic, method)
        ph1_stats = ph1_stats.append(ph1)
        ph2_stats = ph2_stats.append(ph2)
    return ph1_stats.drop(['Total Relevant'], axis=1), ph2_stats.drop(['Train AveP'], axis=1)

def feats_intersect(features):
    common_feats_df = None
    for i, feat in enumerate(features):
        if common_feats_df is None:
            common_feats_df = feat
        else:
            common_feats_df = pd.merge(common_feats_df, feat,
                                       how='inner',
                                       on=['Feature', 'Type'])
    return common_feats_df

def get_most_frequent_features_in_splits(selected_features_per_split):
    selected_features_flat = pd.concat(selected_features_per_split).reset_index(drop=True)
    feature_occurances_series = selected_features_flat.groupby(['Feature', 'Type']).size()
    feature_occurances_df = feature_occurances_series.to_frame('frequency').reset_index()
    feature_occurances_sorted = feature_occurances_df.sort_values(by=['frequency'], ascending=False).reset_index(drop=True)
    
    return feature_occurances_sorted

def summarise_feats(topic, method, k=5):
    feats = load_results_features(topic, method, k)
    return get_most_frequent_features_in_splits(feats)

def plot_phase1_stats(topic):
    methods = utils.get_stat_methods()
    recalls = []
    precs = []
    f1s = [] 
    for method in methods:
        ph1, _ = load_results_stats(topic, method)

        recalls.append(get_confidence_interval_vals(ph1.Recall))
        precs.append(get_confidence_interval_vals(ph1.Precision))
        f1s.append(get_confidence_interval_vals(ph1['F1 Score']))

    start = 0.0
    stop = 1.0
    number_of_lines= len(methods)
    cm_subsection = np.linspace(start, stop, number_of_lines) 
    colors = [ mplt.cm.Dark2(x) for x in cm_subsection ]
    fig = plt.figure(figsize=(15, 5))
    gs = mplt.gridspec.GridSpec(1, 3)
    ax0 = plt.subplot(gs[0, 0])
    x_pos = np.arange(number_of_lines)
    ys, errs = [list(tup) for tup in zip(*recalls)]
    bar1 = ax0.bar(x_pos, ys, yerr=errs, 
            align='center', alpha=0.75, color=colors, edgecolor="black", ecolor='black')
    plt.xticks([], [])
    plt.yticks(fontsize=20)
    plt.ylim(0, 1)
    plt.ylabel("Recall", fontsize=24)
    
    ax1= plt.subplot(gs[0, 1])
    ys, errs = [list(tup) for tup in zip(*precs)]
    ax1.bar(x_pos, ys, yerr=errs, 
            align='center', alpha=0.75, color=colors, edgecolor="black", ecolor='black')
    plt.ylabel("Precision", fontsize=24)
    plt.xticks([], [])
    plt.yticks(fontsize=20)
    plt.ylim(0, 1)

    ax3= plt.subplot(gs[0, 2])
    ys, errs = [list(tup) for tup in zip(*f1s)]
    ax3.bar(x_pos, ys, yerr=errs,
            align='center', alpha=0.75, color=colors, edgecolor="black", ecolor='black')
    plt.ylabel("F1-Score", fontsize=24)
    plt.xticks([], [])
    plt.yticks(fontsize=20)
    plt.ylim(0, 1)

    # #plt.legend([bar1[0], bar1[1]], ['test', 't'])
    plt.tight_layout()
    plt.show()

def plot_ratk_recalls(results_dict, topic, k=18000):
    start = 0.0
    stop = 1.0
    number_of_lines= len(results_dict)
    cm_subsection = np.linspace(start, stop, number_of_lines) 
    colors = [ mplt.cm.Dark2(x) for x in cm_subsection ]    

    fig = plt.figure()
    x_pos = np.arange(number_of_lines)
    ys, errs = [list(tup) for tup in zip(*results_dict.values())]
    plt.bar(x_pos, ys, yerr=errs, 
            align='center',
            width=0.5,
            alpha=0.75,
            color=colors,
            edgecolor="black",
            ecolor='black',
        label=['CILP', 'WILP', 'CAILP', 'Firehose', 'TopK'])


    plt.xticks(x_pos, ['CILP', 'WILP', 'CAILP', 'Firehose', 'TopK'])
    #plt.ylim(0, 1)
    plt.ylabel("R@18,000")
    plt.title("{}".format(utils.get_readable_topic(topic)))
    plt.show()
