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


class Prediction:
    
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
                
        return y_pred, root_tree_building, y_records
    
    
    def Loading_Train_Pred(self,X_train, y_train, X_test, y_test): #Loading training predictions and tree information
        
        y_train_pred, root_tree_building_train, y_records_train = self.Prediction_Train(X_train, y_train)
        
        train_terminal_pred = np.column_stack([y_train_pred, y_records_train[:,-1]])
        train_terminal_pred = np.unique(train_terminal_pred, axis = 0)
        
        y_records = np.zeros((len(y_test),len(root_tree_building_train)))
        y_records[:,0] = np.random.randint(0,1,len(y_test)) + 1   
        
        return y_records, root_tree_building_train, train_terminal_pred, y_train_pred
    
    
    def Test_Node_Assignent(self, X_train, y_train, X_test, y_test): #Assigning tree nodes to test set
        
        y_records, root_tree_building_train, train_terminal_pred, y_train_pred = self.Loading_Train_Pred(X_train, y_train, X_test, y_test)
        
        for k in range(1, len(root_tree_building_train)):
        
            node_cut = root_tree_building_train[k,0]
            cut_value = root_tree_building_train[k,1]
            feature = root_tree_building_train[k,2]
            feature = feature.astype(int)
            
            child_1 = node_cut*2
            child_2 = child_1 + 1
            m = k-1
            
            for i in range(0,len(y_test)):
        
                if y_records[i,m] == node_cut:
                    
                    if X_test[i, feature] < cut_value:
                        y_records[i,k] = child_1
                        
                    else:
                        y_records[i,k] = child_2       
                else:
                    y_records[i,k] = y_records[i,m]
       
        return y_records, train_terminal_pred, y_train_pred
    
       
    def Test_Prediction(self, X_train, y_train, X_test, y_test): #Making predictions on test set
        
        y_records, train_terminal_pred, y_train_pred = self.Test_Node_Assignent(X_train, y_train, X_test, y_test)
   
        y_test_pred = np.zeros((len(y_test))) 
        
        for k in range(0, len(train_terminal_pred)):
            for i in range(0,len(y_records)):
                if y_records[i,-1] == train_terminal_pred[k, 1]:
                    y_test_pred[i] = train_terminal_pred[k,0]
        
        return y_train_pred, y_test_pred
    
    
#-----------------------------------------------------------------------------
    
#Making Prediction on Training and Test Set
Pred = Prediction()
Train_Predictions, Test_Predictions = Pred.Test_Prediction(X_train, y_train, X_test, y_test)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    