# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 18:06:35 2021

@author: maxime + tyler
"""

"""
Test output
"""

#Loading Boston Dataset and Splittng into Training and Test Set

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 


"""
The goal is to do the training and test prediction
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


class Train_Prediction:
    
    def Prediction_Train(self, X, y): #Prediction on Training Set

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
                
        return y_pred
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    