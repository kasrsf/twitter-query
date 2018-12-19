from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer#, CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from CustomTransformers import *
from ClassifierAnalyzer import *
from Filter import *
from Evaluation import *

class TestFramework:
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test
                
        print("Initializing Classifier....")
        self.pipeline = self.initialize_pipeline()
        print("Complete!")
        self.analyzer = ClassifierAnalyzer(self.pipeline)
        
    def initialize_pipeline(self):        
        train_features = self.train.drop('label', axis=1)
        train_target = self.train['label']
        
        validation_features = self.valid.drop('label', axis=1)
        valid_target = self.valid['label']
        
        feature_transformer = FeatureUnion(
                                    transformer_list=[
                                        ('term', Pipeline([
                                            ('selector', ItemSelector(key='term')),
                                            ('count', CountVectorizer(stop_words='english', min_df=10)),
                                        ])),

                                        ('hashtag', Pipeline([
                                            ('selector', ItemSelector(key='hashtag')),
                                            ('count', CountVectorizer(min_df=10))#, lowercase=False
                                        ])),

                                        ('user', Pipeline([
                                            ('selector', ItemSelector(key='from_user')),
                                            ('count', CountVectorizer(min_df=10)),
                                        ])),

                                        ('location', Pipeline([
                                            ('selector', ItemSelector(key='location')),
                                            ('count', CountVectorizer(min_df=10)),
                                        ])),

                                        ('mention', Pipeline([
                                            ('selector', ItemSelector(key='mention')),
                                            ('count', CountVectorizer(min_df=10)),
                                        ]))
                                    ])
            
        train_valid = pd.concat([train_features, validation_features])
        train_valid_target = pd.concat([train_target, valid_target])

        train_valid_transformed = feature_transformer.fit_transform(train_valid)
        
        train_transformed = feature_transformer.transform(train_features)
        valid_transformed = feature_transformer.transform(validation_features)
        
        # select c for best accuracy
        Cs = [0.01, 0.1, 1, 10, 100] 
        best_classifier = None
        best_score = 0
        
        for c in Cs:
            classifier = LogisticRegression(C=c)
            classifier.fit(train_transformed, train_target)    
            c_score = classifier.score(valid_transformed, valid_target)
            if c_score > best_score:
                best_classifier = classifier
                best_score = c_score
        
        #best_classifier = LogisticRegressionCV()
        #best_classifier.fit(train_transformed, train_target)
               
        return Pipeline(steps=[('features', feature_transformer), ('classifier', best_classifier)])

    def get_pipeline(self):
        return self.pipeline
    
    def get_top_features(self, k):
        top_terms, top_hashtags, top_users, top_locs, top_mentions = self.analyzer.get_top_features(k)
        
        print("Top Terms: ", top_terms)
        print("Top Hashtags: ", top_hashtags)
        print("Top Users: ", top_users)
        print("Top Locations: ", top_locs)
        print("Top Mentions: ", top_mentions)
    
    def get_top_features(self, k):
        return self.analyzer.get_top_features(k)
    
    def get_filtered_test_data(self, k, verbose=False):
        top_terms, top_hashtags, top_users, top_locs, top_mentions = self.get_top_features(k)
        
        if verbose == True:
            print("Top Terms: ", top_terms)
            print("Top Hashtags: ", top_hashtags)
            print("Top Users: ", top_users)
            print("Top Locations: ", top_locs)
            print("Top Mentions: ", top_mentions)

        data = filter_data(self.test, top_terms, top_hashtags, top_users, top_locs, top_mentions)
        
        return data
    
    def get_filtered_data_by_index(self, indexes):
        features = get_feature_by_index(self.pipeline, indexes)
        
        terms = features[features.Type == "Term"].Feature.tolist()
        hashtags = features[features.Type == "Hashtag"].Feature.tolist()
        users = features[features.Type == "User"].Feature.tolist()
        locs = features[features.Type == "Location"].Feature.tolist()
        mentions = features[features.Type == "Mention"].Feature.tolist()
        
        data = filter_data(self.test, terms, hashtags, users, locs, mentions)
        
        return data
        
    def get_ideal_performance(self):
        test_features = self.test.drop('label', axis=1)
        test_target = self.test['label']

        predictions = self.pipeline.predict_proba(test_features)[:, 1]
        
        return model_evaluation_summary(test_target, predictions)
    
    def run_config_by_index(self, indexes):
        test = self.get_filtered_data_by_index(indexes)
        
        test_features = test.drop('label', axis=1)
        test_target = test['label']

        predictions = self.pipeline.predict_proba(test_features)[:, 1]
        return model_evaluation_summary(test_target, predictions)
    
    def run_with_filtered_test_data(self, test):        
        test_features = test.drop('label', axis=1)
        test_target = test['label']

        predictions = self.pipeline.predict_proba(test_features)[:, 1]
     
        return model_evaluation_summary(test_target, predictions)
     
    def run_configs(self, configs=[]):
        pipelines = []
        positive_counts = []
        total_counts = []
        aveps = []
        prec_recall = []
        
        for config in configs:
            test = self.get_filtered_test_data(config)
            positive_counts.append(len(test[test.label == 1]))     
            total_counts.append(len(test))
            
            test_features = test.drop('label', axis=1)
            test_target = test['label']
            
            predictions = self.pipeline.predict_proba(test_features)[:, 1]
            aveps.append(average_precision_score(test_target, predictions))

            precision, recall, _ = precision_recall_curve(test_target, predictions)
            prec_recall.append([precision, recall])
            
        return positive_counts, total_counts, pipelines, aveps, prec_recall