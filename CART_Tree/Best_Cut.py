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
    
    def visit_all_features(self, X, y, sample_f=None):
        
        root_purity = self.MeasureOfDispersion(y,[])
        
        self.splits_evaluation = np.array([[len(y),np.nan,root_purity, -1]])    # ensure that previous split is not better
        
        if sample_f == None:
            sample_f = X.shape[1]
            features = np.linspace(0,X.shape[1]-1,X.shape[1]).astype(int)
     
        else:
            sample = min(X.shape[1],sample_f)
            feature_columns = np.linspace(0,X.shape[1]-1,X.shape[1]).astype(int)            
            features = np.random.choice(feature_columns, sample, replace = False)
                
        for f in features:
            
            self.counter_feature_visite += 1
            
            x = X[:,f] # create a vector
            
            cut_value, purity  = self.All_Points(x, y)
            
            x_left = x[x<=cut_value]
            
            cut = len(x_left)
            
            self.splits_evaluation = np.concatenate((self.splits_evaluation, [[cut, cut_value, purity, f]]),0)
            
        # BEST PURITY  
        best_purity = np.min(self.splits_evaluation[:,2])
        binary_filter_puriy = self.splits_evaluation[:,2] == best_purity
        # BEST CUT ON PURITY
        """ it may exist multiple best purity"""
        f_tmp = self.splits_evaluation[binary_filter_puriy,3].min()   
        binary_filter_feature = self.splits_evaluation[:,3] == f_tmp
        # FINAL SELECTED CUT
        binary_filter = np.logical_and(binary_filter_puriy, binary_filter_feature)
        
        if binary_filter.sum() > 1:
            print("ATTENTION MULTIPLE BEST PURITY")
            print(self.splits_evaluation)
        elif binary_filter.sum() ==0:
            print("ATTENTION NO BEST PURITY")
               
        cut_value = self.splits_evaluation[binary_filter,1][0]
        feature = self.splits_evaluation[binary_filter, 3][0]
        
        record = X[:,int(feature)]<=cut_value
        record = abs(record.astype(int)-1)
        
        if feature == -1:
            cut_type = "root"
        else:
            cut_type = "ok"

        return cut, feature, cut_value, cut_type, record
            


"""
test
"""

# BC = Best_cut()


# cut, feature, cut_value, cut_type, record = BC.visit_all_features(X,y)



# X_1 = X[X[:,5]<=6.939,:]
# y_1 = y[X[:,5]<=6.939]
# cut, feature, cut_value, cut_type, record = BC.visit_all_features(X_1,y_1)


# X_2 = X_1[X_1[:,12]<=14.37,:]
# y_2 = y_1[X_1[:,12]<=14.37]
# cut, feature, cut_value, cut_type, record = BC.visit_all_features(X_2,y_2)




















