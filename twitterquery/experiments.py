from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
import time

import classification
import data
import utils

def run_experiment(train_data, train_target, 
                   test_data, test_target,
                   filtering_feature_index):
    # 1nd stage: filtering
    print("begin filtering")
    start_time = time.time()
    filtered_train_data, filtered_train_labels = data.filter_matrix_by_index(train_data,
                                                                            train_target,
                                                                            filtering_feature_index)        
    print("filtering done in {:.2f}s".format(time.time() - start_time))
    phase1_stats = utils.get_labeled_data_statistics(filtered_train_labels)
    # 2nd stage: training
    print("training classifier")
    start_time = time.time()
    classifier = SGDClassifier(loss='log', class_weight='balanced', penalty='elasticnet')
    classifier.fit(filtered_train_data, filtered_train_labels)        
    preds_proba = classifier.predict_proba(train_data)[:, 1]
    train_avep = average_precision_score(train_target, preds_proba)
    preds_proba = classifier.predict_proba(test_data)[:, 1]
    test_avep = average_precision_score(test_target, preds_proba)
    test_patk = classification.p_at_k_score(test_target, preds_proba, k=100)
    test_ratk = classification.r_at_k_score(test_target, preds_proba, k=100)
    print("classifier done in {:.2f} seconds".format(time.time() - start_time))
    return phase1_stats, train_avep, test_avep, test_patk

def run_ratk_experiment(test_data, test_target,
                        filtering_feature_index,
                        n=18000):
    _, filtered_labels = data.filter_matrix_by_index(test_data,
                                                    test_target,
                                                    filtering_feature_index)        
    positive_at_ns = data.count_most_recent_topcial_in_n(filtered_labels, n=n)
    total_n = len(filtered_labels) if len(filtered_labels) < n else n 
    return positive_at_ns, total_n