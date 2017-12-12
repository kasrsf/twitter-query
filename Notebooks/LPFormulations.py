from gurobipy import *
import time 
from sklearn.preprocessing import binarize


def max_cover(feature_coverage, num_tweets, k=20):
    # feature_coverage: tweets covered by each feature
    num_features = len(feature_coverage)

    m = Model()

    f = {} # Binary variable for each feature
    t = {} # Binary variables for each tweet
    
    for i in range(num_features):
        f[i] = m.addVar(vtype=GRB.BINARY, name="f%d" % i)

    for j in range(num_tweets):
        t[j] = m.addVar(vtype=GRB.BINARY, name="t%d" % j)

    m.update()

    for j in range(num_tweets):
        m.addConstr(quicksum(f[i] for i in range(num_features) if j in feature_coverage[i]) >= t[j])

    m.addConstr(quicksum(f[i] for i in range(num_features)) <= k)

    m.setObjective(quicksum(t[j] for j in range(num_tweets)), GRB.MAXIMIZE)

    m.optimize()
    
    model_vars = m.getVars()[:num_features]
    selected_features_index = []

    for i in range(len(model_vars)):
        if model_vars[i].X > 0:
            selected_features_index.append(i)
            
    return selected_features_index
    
def max_cover_with_mutual_information(feature_coverage, mutual_info_scores, num_tweets, mi_weight=1000, k=20):
    # feature_coverage: tweets covered by each feature
    num_features = len(feature_coverage)

    m = Model()
    m.setParam( 'OutputFlag', False )
    
    f = {} # Binary variable for each feature
    t = {} # Binary variables for each tweet
    
    for i in range(num_features):
        f[i] = m.addVar(vtype=GRB.BINARY, name="f%d" % i)

    for j in range(num_tweets):
        t[j] = m.addVar(vtype=GRB.BINARY, name="t%d" % j)

    m.update()

    for j in range(num_tweets):
        m.addConstr(quicksum(f[i] for i in range(num_features) if j in feature_coverage[i]) >= t[j])

    m.addConstr(quicksum(f[i] for i in range(num_features)) <= k)

    m.setObjective(quicksum(t[j] for j in range(num_tweets)) \
                   + (mi_weight * quicksum(f[j]*mutual_info_scores[j] for j in range(num_features))), GRB.MAXIMIZE)

    m.optimize()
    
    model_vars = m.getVars()[:num_features]
    selected_features_index = []

    for i in range(len(model_vars)):
        if model_vars[i].X > 0:
            selected_features_index.append(i)
            
    return selected_features_index

def max_cover_with_mutual_information_normalized(feature_coverage, mutual_info_scores, num_tweets, num_positives, k=20):
    # feature_coverage: tweets covered by each feature
    num_features = len(feature_coverage)

    m = Model()
    #m.setParam('OutputFlag', False )
    
    f = {} # Binary variable for each feature
    t = {} # Binary variables for each tweet
    
    for i in range(num_features):
        f[i] = m.addVar(vtype=GRB.BINARY, name="f%d" % i)

    for j in range(num_tweets):
        t[j] = m.addVar(vtype=GRB.BINARY, name="t%d" % j)

    m.update()

    for j in range(num_tweets):
        m.addConstr(quicksum(f[i] for i in range(num_features) if j in feature_coverage[i]) >= t[j])

    m.addConstr(quicksum(f[i] for i in range(num_features)) <= k)

    m.setObjective((quicksum(t[j] for j in range(num_tweets)) / (float)(num_positives)) \
                   +  quicksum(f[j]*mutual_info_scores[j] for j in range(num_features)), GRB.MAXIMIZE)

    m.optimize()
    
    model_vars = m.getVars()[:num_features]
    selected_features_index = []

    for i in range(len(model_vars)):
        if model_vars[i].X > 0:
            selected_features_index.append(i)
            
    return selected_features_index

def max_cover_with_negs(positive_coverage, negative_coverage, num_positive_tweets, num_negative_tweets\
                        , penalty=0.1, k=20):
    # feature_coverage: tweets covered by each feature
    num_features = len(positive_coverage)

    m = Model()
    m.setParam('OutputFlag', False )
    
    f = {} # Binary variable for each feature
    p = {} # Binary variables for each topical(positive) tweet
    n = {} # Binary variables for each non-topical(negative) tweet
    
    for i in range(num_features):
        f[i] = m.addVar(vtype=GRB.BINARY, name="f%d" % i)

    for j in range(num_positive_tweets):
        p[j] = m.addVar(vtype=GRB.BINARY, name="p%d" % j)
        
    for j in range(num_negative_tweets):
        n[j] = m.addVar(vtype=GRB.BINARY, name="n%d" % j)
   
    m.update()

    for j in range(num_positive_tweets):
        m.addConstr(quicksum(f[i] for i in range(num_features) if j in positive_coverage[i]) >= p[j])

    for j in range(num_negative_tweets):
        m.addGenConstrOr(n[j], [f[i] for i in range(num_features) if j in negative_coverage[i]])        
        
    m.addConstr(quicksum(f[i] for i in range(num_features)) <= k)

    m.setObjective(quicksum(p[i] for i in range(num_positive_tweets)) \
                   - (penalty * (quicksum(n[i] for i in range(num_negative_tweets)))), GRB.MAXIMIZE)
    m.optimize()
    
    model_vars = m.getVars()[:num_features]
    selected_features_index = []

    for i in range(len(model_vars)):
        if model_vars[i].X > 0:
            selected_features_index.append(i)
            
    return selected_features_index

def max_cover_with_negs_unweighted(positive_coverage, negative_coverage, num_positive_tweets, num_negative_tweets\
                        , k=20, time_limit_secs=None, print_elapsed=False):
    # feature_coverage: tweets covered by each feature
    num_features = len(positive_coverage)

    if print_elapsed == True:
        start_time = time.time()
   
    m = Model()
    m.setParam('OutputFlag', False)
    
    if time_limit_secs != None:
        m.setParam('TimeLimit', time_limit_secs)
    
    f = {} # Binary variable for each feature
    p = {} # Binary variables for each topical(positive) tweet
    n = {} # Binary variables for each non-topical(negative) tweet
    
    for i in range(num_features):
        f[i] = m.addVar(vtype=GRB.BINARY, name="f%d" % i)

    for j in range(num_positive_tweets):
        p[j] = m.addVar(vtype=GRB.BINARY, name="p%d" % j)
        
    for j in range(num_negative_tweets):
        n[j] = m.addVar(vtype=GRB.BINARY, name="n%d" % j)
   
    m.update()

    for j in range(num_positive_tweets):
        m.addConstr(quicksum(f[i] for i in range(num_features) if j in positive_coverage[i]) >= p[j])
    for j in range(num_negative_tweets):
        m.addGenConstrOr(n[j], [f[i] for i in range(num_features) if j in negative_coverage[i]])        

    m.addConstr(quicksum(f[i] for i in range(num_features)) <= k)


    m.setObjective(quicksum(p[i] for i in range(num_positive_tweets)) \
                   - ((quicksum(n[i] for i in range(num_negative_tweets)))), GRB.MAXIMIZE)
    
    m.optimize()
    
    model_vars = m.getVars()[:num_features]
    selected_features_index = []

    for i in range(len(model_vars)):
        if model_vars[i].X > 0:
            selected_features_index.append(i)
            
    if print_elapsed == True:
        print("Running Time: {0} Seconds".format(time.time() - start_time))
            
    return selected_features_index

def greedy_max_cover(positive_set, negative_set, pipeline, k=20, print_solve_time=False):
    num_positive_tweets = positive_set.shape[0]
    num_negative_tweets = negative_set.shape[0]
    num_features = positive_set.shape[1]
      
    positive_bin = binarize(positive_set)
    negative_bin = binarize(negative_set)

    positive_lil = positive_bin.tolil()
    negative_lil = negative_bin.tolil()
    
    if print_solve_time == True:
        start_time = time.time()

    selected_features = []

    for i in range(k):
        max_score = -99999
        selected_feature = -1

        scores = positive_lil.sum(axis=0) - negative_lil.sum(axis=0)
        selected_feature = scores.argmax()

        if selected_feature not in selected_features:
            selected_features.append(selected_feature)
            positive_lil[:, selected_feature] = 0
            negative_lil[:, selected_feature] = 0
        else:
            break
    
    if print_solve_time == True:
        print("Running Time: {0} Seconds".format(time.time() - start_time))
    
    return selected_features
