from FeatureExtraction import *

class ClassifierAnalyzer():
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def get_top_features(self, k=[20, 20, 20, 20, 20]):
        return top_features_by_kind(self.pipeline, k)
    
    def print_top_features(self, k=[20, 20, 20, 20, 20]):
        top_terms, top_hashtags, top_users, top_locs, top_mentions = self.get_top_features(k)
        
        if k[0] > 0:
            print("Top Terms: " + str(top_terms).strip('[]') + "\n")
        if k[1] > 0:
            print("Top Hashtags: " + str(top_hashtags).strip('[]') + "\n")
        if k[2] > 0:
            print("Top Users: " + str(top_users).strip('[]') + "\n")            
        if k[3] > 0:
            print("Top Locations: " + str(top_locs).strip('[]') + "\n")            
        if k[4] > 0:
            print("Top Mentions: " + str(top_mentions).strip('[]') + "\n")        