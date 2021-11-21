#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:24:44 2021
@author: maxime
"""

"""
Test output
"""

# Load data
#from sklearn.datasets import load_boston
#X, y = load_boston(return_X_y=True)


"""
The goal of this class is to create a loop to visit all the features and find the best split.
IMPORTANT
IMPORTANT
In HERE, we implement the Random Forest strategy of sampling (features and observations) IMPORTANT
IMPORTANT
IMPORTANT
We also define here the type of split strategy.
Discuss about: ensuring that previous split is not better
Input:
    - X is a matrix of features
    - y is a response vector
    
Output:
    - best cut
"""

# Libraries
import Split_Search as split
import numpy as np


class Best_cut(split.Best_Splitting_Point):
    
    def __init__ (self):
        self.counter_feature_visite = 0
        split.Best_Splitting_Point.__init__(self)
    
    def visit_all_features(self, X, y, num_feat=None):
        
        root_purity = self.MeasureOfDispersion(y,[])
        
        best_feature = -1
        best_cut_value = np.nan
        best_purity = root_purity  
        best_cut = len(y)
        
        if num_feat == None:
            num_feat = X.shape[1]
            features = np.linspace(0,X.shape[1]-1,X.shape[1]).astype(int)
     
        else:
            sample = min(X.shape[1],num_feat)
            feature_columns = np.linspace(0,X.shape[1]-1,X.shape[1]).astype(int)            
            features = np.random.choice(feature_columns, sample, replace = False)
                
        for f in features:
            
            self.counter_feature_visite += 1
            
            x = X[:,f] # create a vector
            
            cut_value, purity  = self.All_Points(x, y)
                       
            if purity < best_purity:
                best_purity = purity
                best_cut_value = cut_value
                best_feature = f
                best_cut = len(y[x<=cut_value])
            
        record = X[:,int(best_feature)]<=best_cut_value
        record = abs(record.astype(int)-1)
        
        if best_feature == -1:
            cut_type = "root"
        else:
            cut_type = "ok"

        return best_cut, best_feature, best_cut_value, cut_type, record
            


"""
test
"""

#BC = Best_cut()


#cut, feature, cut_value, cut_type, record = BC.visit_all_features(X,y)

#t = BC.splits_evaluation
#x1 = BC.x
#y1 = BC.y
# X_1 = X[X[:,5]<=6.939,:]
# y_1 = y[X[:,5]<=6.939]
# cut, feature, cut_value, cut_type, record = BC.visit_all_features(X_1,y_1)


# X_2 = X_1[X_1[:,12]<=14.37,:]
# y_2 = y_1[X_1[:,12]<=14.37]
# cut, feature, cut_value, cut_type, record = BC.visit_all_features(X_2,y_2)




















