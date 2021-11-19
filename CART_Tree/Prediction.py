# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:13:51 2021

@author: tyler
"""


import numpy as np
import time

class Prediction:
    
    def Predict(self, X_test, y, L_records, L_root_tree_building, L_train_pred):
        
        start = time.time()
        print("Start")
        
        y_pred = self.Loop_Pred(X_test, y, L_records, L_root_tree_building, L_train_pred)
        
        end = time.time()
        print("Elalpsed time: ",str(end - start), "Seconds")
        
        return y_pred
                    
    def Loop_Pred(self, X, y, L_records, L_root_tree_building, L_train_pred):
        
        y_pred = []
        
        for v in range(0, len(L_records)-1):
        
                    y_records_train = L_records[v]
                    root_tree_building_train = L_root_tree_building[v]
                    y_train_pred = L_train_pred[v]
                    
                    train_terminal_pred = np.column_stack([y_train_pred, y_records_train[:,-1]])
                    train_terminal_pred = np.unique(train_terminal_pred, axis = 0)
                    
                    y_records = np.zeros((len(y),len(root_tree_building_train)))
                    y_records[:,0] = np.random.randint(0,1,len(y)) + 1  
      
                    
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
               
                    y_test_pred = np.zeros((len(y))) 
                    
                    #Simplify function ()
                    for m in range(0, len(train_terminal_pred)):
                        for j in range(0,len(y_records)):
                            if y_records[j,-1] == train_terminal_pred[m, 1]:
                                y_test_pred[j] = train_terminal_pred[m,0]
                    
                    y_pred.append(y_test_pred)
                    
        y_pred = np.transpose(np.vstack(y_pred))
        y_pred = y_pred.mean(axis=1) 
        
        return y_pred

#Sanity check to see if middle boolean function
#np.unique()











    
