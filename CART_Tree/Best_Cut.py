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
import Measure_Purity as sim

MoD = sim.MeasureOfDispersion()


class Best_cut:
    
    def visit_all_features(self, X, y):
        
        
        root_purity = MoD.MeasureOfDispersion(y,[])
        
        splits_evaluation = np.array([[len(y),np.nan,root_purity, -1]])    # ensure that previous split is not better
        
        for f in range(0,X.shape[1]):
            
            x = X[:,f] # create a vector
            
            feature_split = split.Best_Splitting_Point(x, y) # load the class --> should be load once!    
                       
            cut_value, purity  = feature_split.All_Points() # get cut and purity
            
            x_left = x[x<=cut_value]
            
            cut = len(x_left)
            
            splits_evaluation = np.concatenate((splits_evaluation, [[cut, cut_value, purity, f]]),0)
            
        
        best_purity = np.min(splits_evaluation[:,2])
        cut_value = splits_evaluation[splits_evaluation[:,2] == best_purity,1][0]
        cut = splits_evaluation[splits_evaluation[:,2] == best_purity,0][0]
        feature = splits_evaluation[splits_evaluation[:,0] == cut, 3][0]

        return cut, feature, cut_value
            


"""
test
"""

#BC = Best_cut()
#cut, feature, cut_value = BC.visit_all_features(X_train,y_train)




























