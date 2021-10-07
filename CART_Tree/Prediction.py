#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 14:11:31 2021

@author: maxime
"""


"""
Test output
"""

# Load data
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)


"""
The goal is to do the training prediction

!! for test prediction, WE NEED TO RECORD THE FEATURES AND THE CUT!!

We need to apply majority vote for the boolean and average for the real values.

In order to know the majority, we can simply sum the boolean values (sum the ones) and if they are the majority, we apply the value
one otherwise we apply the value zero.

Input:
    - X is a matrix of features
    - y is a response vector
    
Output:
    - predictions
"""

import Nodes_Creation as nc
import numpy as np


class Prediction:
    
    def prediction_train(self, X, y):

        NS = nc.NodeSearch()
        
        y_records, root_tree_building = NS.breath_first_search(X, y, min_bucket=5, max_size = 4)
        
        y_last = y_records[:,y_records.shape[1]-1]
        y_pred = y.copy()
        
        tree_nodes = np.unique(y_last)
        
        for n in tree_nodes:
            print(n)
            
            if y.dtype == 'bool':  # if boolean --> majority vote
                sum_pred = y[y_pred==n].sum()   # sum = value of the yes
                len_pred = len(y[y_pred==n])    # size of the pool
                
                if sum_pred > len_pred/2:       # if value of yes are mojority then 1 otherwise 0
                    y_pred[y_pred==n] = 1
                else: y_pred[y_pred==n] = 0
                
            else:    
                y_pred[y_last==n] = y[y_last==n].mean()
                
        return y_pred, root_tree_building
    
y_pred, root_tree_building = Prediction().prediction_train(X,y)

print("The train predicitons are....")
print(y_pred)

