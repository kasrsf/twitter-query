import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve

# from CustomTransformers import *
# from ClassifierAnalyzer import *
# from Filter import *

def model_evaluation_summary(target, prediction, k=1000):
    p_at_k = p_at_k_score(target, prediction, k)
    r_at_k = r_at_k_score(target, prediction, k)
    avep = average_precision_score(target, prediction) 
    
    precision, recall, _ = precision_recall_curve(target, prediction)
    prec_recall = [precision, recall]
    
    return p_at_k, r_at_k, avep, prec_recall

def print_evaluation_summary(target, prediction , k=100):
    p, r, a = model_evaluation_summary(target, prediction, k)
    print("Precision@{0} = {1} \nRecall@{0} = {2}\nAveP = {3}".format(k, p, r, a))