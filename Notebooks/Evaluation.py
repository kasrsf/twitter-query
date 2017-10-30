import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score
import numpy as np

def p_at_k_score(y_true, y_score, k):
    df = pd.DataFrame({'true': y_true.tolist(), 'score': y_score.tolist()})\
                    .sort_values(by=['score'], ascending = False)[:k]

    THRESHOLD = 0.5
    df['pred'] = [1 if i > THRESHOLD else 0 for i in df['score']]
    topk = df.iloc[:k, ]

    return precision_score(topk.loc[:, 'true'], topk.loc[:, 'pred'])

def r_at_k_score(y_true, y_score, k):
    df = pd.DataFrame({'true': y_true.tolist(), 'score': y_score.tolist()})\
                    .sort_values(by=['score'], ascending = False)[:k]

    THRESHOLD = 0.5
    df['pred'] = [1 if i > THRESHOLD else 0 for i in df['score']]
    topk = df.iloc[:k, ]

    return recall_score(topk.loc[:, 'true'], topk.loc[:, 'pred'])

def model_evaluation_summary(target, prediction, k=100):
    p_at_k = p_at_k_score(target, prediction, k)
    r_at_k = r_at_k_score(target, prediction, k)
    avep = average_precision_score(target, prediction) 
    
    return p_at_k, r_at_k, avep

def print_evaluation_summary(target, prediction , k=100):
    p, r, a = model_evaluation_summary(target, prediction, k)
    print("Precision@{0} = {1} \nRecall@{0} = {2}\nAveP = {3}".format(k, p, r, a))