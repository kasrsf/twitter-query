import gurobipy as gurobi
import time 
from sklearn.preprocessing import binarize
import numpy as np

import utils

def gurobi_cilp(tweet_feature_matrix, k=20):
    num_tweets, num_features = tweet_feature_matrix.shape
    m = gurobi.Model()
    m.setParam('OutputFlag', False )
    # binary variable for each feature
    f = {}
    # binary variables for each tweet 
    t = {}
    for i in range(num_features):
        f[i] = m.addVar(vtype=gurobi.GRB.BINARY, name="f%d" % i)
    for j in range(num_tweets):
        t[j] = m.addVar(vtype=gurobi.GRB.BINARY, name="t%d" % j)
    m.update()
    for j in range(num_tweets):
        tweet_feats = tweet_feature_matrix[j].nonzero()[1]
        m.addConstr(gurobi.quicksum(f[i] for i in tweet_feats) >= t[j])
    m.addConstr(gurobi.quicksum(f[i] for i in range(num_features)) <= k)
    m.setObjective(gurobi.quicksum(t[j] for j in range(num_tweets)), gurobi.GRB.MAXIMIZE)
    print("model built")
    m.optimize()
    model_vars = m.getVars()[:num_features]
    selected_features_index = []
    for i in range(len(model_vars)):
        if model_vars[i].X > 0:
            selected_features_index.append(i)
    return selected_features_index

def greedy_cilp(positive_set, k=20):
    num_positive_tweets, num_features = positive_set.shape
    positive_bin = binarize(positive_set)
    positive_lil = positive_bin.tolil()
    selected_features = []
    for i in range(k):
        scores = positive_lil.sum(axis=0)
        selected_feature = scores.argmax()
        if selected_feature not in selected_features:
            covered_tweets = list(positive_lil[:, selected_feature].nonzero()[0])
            utils.delete_row_lil(positive_lil, covered_tweets)
            selected_features.append(selected_feature)
        else:
            break
    return selected_features
    
def gurobi_wilp(tweet_feature_matrix, mutual_info_scores, mi_weight=1000, k=20):
    num_tweets, num_features = tweet_feature_matrix.shape
    m = gurobi.Model()
    m.setParam('OutputFlag', False )
    feature_selected = {} 
    tweets_covered = {} 
    for i in range(num_features):
        feature_selected[i] = m.addVar(vtype=gurobi.GRB.BINARY, name="f%d" % i)
    for j in range(num_tweets):
        tweets_covered[j] = m.addVar(vtype=gurobi.GRB.BINARY, name="t%d" % j)
    m.update()
    for j in range(num_tweets):
        tweet_feats = tweet_feature_matrix[j].nonzero()[1]
        m.addConstr(gurobi.quicksum(feature_selected[i] for i in tweet_feats) >= tweets_covered[j])
    m.addConstr(gurobi.quicksum(feature_selected[i] for i in range(num_features)) <= k)
    m.setObjective((gurobi.quicksum(tweets_covered[j] for j in range(num_tweets)) / float(num_tweets)) 
                   + (mi_weight * gurobi.quicksum(feature_selected[j] * mutual_info_scores[j] for j in range(num_features))), gurobi.GRB.MAXIMIZE)
    m.optimize()
    model_vars = m.getVars()[:num_features]
    selected_features_index = []
    for i in range(len(model_vars)):
        if model_vars[i].X > 0:
            selected_features_index.append(i)
    return selected_features_index

def greedy_wilp(positive_set, mutual_info_scores, mi_weight=1, k=20):
    num_tweets, num_features = positive_set.shape      
    positive_bin = binarize(positive_set)
    positive_lil = positive_bin.tolil()
    mi_scores_mat = np.matrix(mutual_info_scores)
    selected_features = []
    for i in range(k):
        scores = (positive_lil.sum(axis=0) / float(num_tweets)) + (mi_weight * mi_scores_mat)
        selected_feature = scores.argmax()
        if selected_feature not in selected_features:
            covered_tweets = list(positive_lil[:, selected_feature].nonzero()[0])
            utils.delete_row_lil(positive_lil, covered_tweets)
            selected_features.append(selected_feature)
            mi_scores_mat[0, selected_feature] = 0
        else:
            break    
    return selected_features 

def gurobi_cailp(positive_coverage, negative_coverage, k=20):
    # feature_coverage: tweets covered by each feature
    num_features = positive_coverage.shape[1]
    num_positive_tweets = positive_coverage.shape[0]
    if num_positive_tweets > 1000:
        positive_coverage, _ = utils.sample_sparse_matrix(positive_coverage, n=1000)
        num_positive_tweets = positive_coverage.shape[0]
    negative_coverage, _ = utils.sample_sparse_matrix(negative_coverage, n=int(num_positive_tweets))
    num_negative_tweets = negative_coverage.shape[0]

    m = gurobi.Model()
    m.setParam('OutputFlag', False )    
    feature_selected = {} 
    positive_tweet_covered = {}
    negative_tweet_covered = {}
    for i in range(num_features):
        feature_selected[i] = m.addVar(vtype=gurobi.GRB.BINARY, name="f%d" % i)
    for j in range(num_positive_tweets):
        positive_tweet_covered[j] = m.addVar(vtype=gurobi.GRB.BINARY, name="p%d" % j)
    for j in range(num_negative_tweets):
        negative_tweet_covered[j] = m.addVar(vtype=gurobi.GRB.BINARY, name="n%d" % j)
    m.update()
    for j in range(num_positive_tweets):
        tweet_feats = positive_coverage[j].nonzero()[1]
        m.addConstr(gurobi.quicksum(feature_selected[i] for i in tweet_feats) >= positive_tweet_covered[j])
    for j in range(num_negative_tweets):
        tweet_feats = negative_coverage[j].nonzero()[1]
        m.addGenConstrOr(negative_tweet_covered[j], [feature_selected[i] for i in tweet_feats])  
    m.addConstr(gurobi.quicksum(feature_selected[i] for i in range(num_features)) <= k)
    m.setObjective((gurobi.quicksum(positive_tweet_covered[i] for i in range(num_positive_tweets)) / float(num_positive_tweets))
                   - (gurobi.quicksum(negative_tweet_covered[i] for i in range(num_negative_tweets)) / float(num_negative_tweets))
                   , gurobi.GRB.MAXIMIZE)
    m.optimize()
    model_vars = m.getVars()[:num_features]
    selected_features_index = []
    for i in range(len(model_vars)):
        if model_vars[i].X > 0:
            selected_features_index.append(i)    
    return selected_features_index

def greedy_cailp(positive_coverage, negative_coverage, k=20):
    num_features = positive_coverage.shape[1]
    num_positive_tweets = positive_coverage.shape[0]
    num_negative_tweets = negative_coverage.shape[0]      
    positive_bin = binarize(positive_coverage)
    negative_bin = binarize(negative_coverage)
    positive_lil = positive_bin.tolil()
    negative_lil = negative_bin.tolil()
    selected_features = []
    for i in range(k):
        print(i)
        scores = (positive_lil.sum(axis=0) / float(num_positive_tweets)) \
                - (negative_lil.sum(axis=0) / float(num_negative_tweets))
        selected_feature = scores.argmax()
        if selected_feature not in selected_features:
            covered_pos_tweets = list(positive_lil[:, selected_feature].nonzero()[0])
            utils.delete_row_lil(positive_lil, covered_pos_tweets)
            covered_neg_tweets = list(negative_lil[:, selected_feature].nonzero()[0])
            utils.delete_row_lil(negative_lil, covered_neg_tweets)
            selected_features.append(selected_feature)
        else:
            break
    return selected_features
