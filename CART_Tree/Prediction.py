# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:13:51 2021

@author: tyler
"""


import numpy as np
import time

class Prediction:
    
    def Predict(self, X_test, y_test, L_records, L_root_tree_building, L_train_pred):
        
        start = time.time()
        print("Start")
        
        y_pred = self.Loop_Pred(X_test, y_test, L_records, L_root_tree_building, L_train_pred)
        
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
                        
                        for i in range(0,len(y)):
                    
                            if y_records[i,m] == node_cut:
                                
                                if X[i, feature] <= cut_value:
                                    y_records[i,k] = child_1
                                    
                                else:
                                    y_records[i,k] = child_2       
                            else:
                                y_records[i,k] = y_records[i,m]
               
                    y_test_pred = np.zeros((len(y))) 
                    
                    for m in range(0, len(train_terminal_pred)):
                        for j in range(0,len(y_records)):
                            if y_records[j,-1] == train_terminal_pred[m, 1]:
                                y_test_pred[j] = train_terminal_pred[m,0]
                    
                    y_pred.append(y_test_pred)
                    
        y_pred = np.transpose(np.vstack(y_pred))
        y_pred = y_pred.mean(axis=1) 
        
        return y_pred

    
