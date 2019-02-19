from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
import numpy as np
import pandas as pd

def get_classifier_top_weighted_features(classifier, k=50):
    return np.flipud(np.argsort(classifier.coef_[0])[-k:])

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