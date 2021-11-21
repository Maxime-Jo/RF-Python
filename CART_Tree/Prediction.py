# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:13:51 2021

@author: tyler
"""


import numpy as np
import time
from joblib import Parallel, delayed

class Prediction:
    
    def Predict_y(self, X_test, L_records, L_root_tree_building, L_train_pred, n_core=1):
                
        print("Start")
        start = time.time()

        if n_core == 1: 
            y_pred = self.Loop_Pred(X_test, L_records, L_root_tree_building, L_train_pred)
        else:
            y_pred = self.Parallel_Pred(X_test, L_records, L_root_tree_building, L_train_pred, n_core)
        
        end = time.time()
        print("Prediction Elalpsed time: ",str(end - start), "Seconds")
        
        return y_pred
    
    def Parallel_Pred(self, X, L_records, L_root_tree_building, L_train_pred, n_core=1):
        
        test_pred = Parallel(n_jobs=n_core)(delayed(self.Test_Pred)(X, L_records[i], L_root_tree_building[i], L_train_pred[i]) for i in range(len(L_records)))
                                            
        y_pred = []
        for y_test_pred in test_pred:
            y_pred.append(y_test_pred)
                    
        y_pred = np.transpose(np.vstack(y_pred))
        y_pred = y_pred.mean(axis=1) 

        return y_pred            
    
    def Test_Pred(self, X, y_records_train, root_tree_building_train, y_train_pred):
        
        train_terminal_pred = np.column_stack([y_train_pred, y_records_train[:,-1]])
        train_terminal_pred = np.unique(train_terminal_pred, axis = 0)
        
        y_records = np.zeros((len(X),len(root_tree_building_train)))
        y_records[:,0] = np.random.randint(0,1,len(X)) + 1  
  
        
        for k in range(1, len(root_tree_building_train)):
        
            node_cut = root_tree_building_train[k,0]
            cut_value = root_tree_building_train[k,1]
            feature = root_tree_building_train[k,2]
            feature = feature.astype(int)
            
            child_1 = node_cut*2
            child_2 = child_1 + 1
            m = k-1
            
            boolean_child_1 = np.logical_and(X[:, feature] <= cut_value,
                                     y_records[:,m]==node_cut)
            boolean_child_2 = np.logical_and(X[:, feature] > cut_value,
                                     y_records[:,m]==node_cut)
            
            y_records[:,k] =  y_records[:,m]
            y_records[boolean_child_1, k] = child_1
            y_records[boolean_child_2, k] = child_2
   
        y_test_pred = np.zeros((len(X))) + float('inf')
        y_records_last = y_records[:,-1]
        
        for pred in train_terminal_pred:
            y_test_pred[y_records_last==pred[1]] = pred[0]
 
        return (y_test_pred)
                    
    def Loop_Pred(self, X, L_records, L_root_tree_building, L_train_pred):
        
        y_pred = []
        
        for v in range(0, len(L_records)):
        
            y_records_train = L_records[v]
            root_tree_building_train = L_root_tree_building[v]
            y_train_pred = L_train_pred[v]
            
            y_test_pred = self.Test_Pred(X, y_records_train, root_tree_building_train, y_train_pred)
   
            y_pred.append(y_test_pred)
                    
        y_pred = np.transpose(np.vstack(y_pred))
        y_pred = y_pred.mean(axis=1) 
        
        return y_pred





