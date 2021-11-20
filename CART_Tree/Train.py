# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:45:24 2021

@author: tyler + maxime
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 18:06:35 2021
@author: maxime + tyler
"""

"""
Test output
"""

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

# Load data
# from sklearn.datasets import load_boston
# X, y = load_boston(return_X_y=True)

import Nodes_Creation as nc
import numpy as np
import Reduce_n_complexity as rc
import multiprocessing
from joblib import Parallel, delayed



class Train(rc.Reduce_Complexity,nc.NodeSearch) :
    
    def __init__ (self):
        self.counter_tree_visite = 0
        rc.Reduce_Complexity.__init__(self)
        nc.NodeSearch.__init__(self)
    
    
    def CART_Train(self,X, y, num_feat = None, 
                 sample_n = None,
                 min_bucket=5, max_size = 6, 
                 strategy=None, bins = None):
        
        #RC = rc.Reduce_Complexity()
        #NS = nc.NodeSearch()
        
        self.counter_tree_visite += 1
        
        """ Bootstrap N"""
        if sample_n == None:
            observations_sample = np.linspace(0,X.shape[0]-1,X.shape[0]).astype(int)
        else:
            sample = int(min(sample_n,1)*X.shape[0])
            observations = np.linspace(0,X.shape[0]-1,X.shape[0]).astype(int)            
            observations_sample = np.random.choice(observations, sample, replace = True)
            
        """ Sample """
        X_sample = X[observations_sample,:]
        y_sample = y[observations_sample]
        
        """ Reduce Complexity"""
        #X_sample = RC.reduce(X_sample,bins,strategy)
        X_sample = self.reduce(X_sample,bins,strategy)
        
        """ Get tree """
        #y_records, root_tree_building = NS.breath_first_search(X_sample, y_sample, min_bucket=min_bucket,
        #                                                       max_size = max_size, sample_f=sample_f)
        y_records, root_tree_building = self.breath_first_search(X_sample, y_sample, min_bucket=min_bucket,
                                                               max_size = max_size, num_feat=num_feat)
        
        y_pred = self.Tree_Prediction(y_records, y_sample)
        
        return y_records, root_tree_building, y_pred
    
    
    def Tree_Prediction(self, y_records, y):
        
        y_last = y_records[:,y_records.shape[1]-1]
        y_pred = y.copy()
        
        tree_nodes = np.unique(y_last)
                
        for n in tree_nodes:
            print(n)
                        
            if y.dtype == 'bool':  # if boolean --> majority vote
                sum_pred = y[y_pred==n].sum()   # sum = value of the yes
                len_pred = len(y[y_pred==n])    # size of the pool
            
                if sum_pred > len_pred/2:       # if value of yes are majority then 1 otherwise 0
                    y_pred[y_pred==n] = 1
            
                else: y_pred[y_pred==n] = 0
            
            else:    
                y_pred[y_last==n] = y[y_last==n].mean()
        
        return y_pred      
    
    def RF_Train(self, X, y, num_feat = 3, 
                 n_tree = 1, sample_n = None,
                 min_bucket=1, max_size = 6, cores = 1, 
                 strategy=None, bins = None):
        
        L_records = []
        L_root_tree_building = []
        L_train_pred = []
        
        cores = min(cores, multiprocessing.cpu_count()-1)
        
        if cores == 1:     
            for n in range(0,n_tree):
                    print("#######################")
                    print("TREE: "+str(n))
                    print("#######################")
                    
                    
                    """ Get tree """
                    y_records, root_tree_building, y_pred = self.CART_Train(X, y, num_feat = num_feat, 
                                                                    sample_n = sample_n,
                                                                    min_bucket=min_bucket, max_size = max_size,
                                                                    strategy=strategy, bins = bins)
                
                    """ Append results """
                    L_records.append(y_records)
                    L_root_tree_building.append(root_tree_building)
                    L_train_pred.append(y_pred)
                    
            
        else:            
            processed_list = Parallel(n_jobs=cores)(delayed(self.CART_Train)(X, y, num_feat = num_feat, 
                                                                    sample_n = sample_n,
                                                                    min_bucket=min_bucket, max_size = max_size,
                                                                    strategy=strategy, bins = bins) for i in range(0,n_tree))
            
            for tree in processed_list:
                y_records, root_tree_building, y_pred = tree
    
                """ Append results """
                L_records.append(y_records)
                L_root_tree_building.append(root_tree_building)
                L_train_pred.append(y_pred)
                
                
        print("#######################")
        print("TRAINING DONE")
        print("#######################")
        
        # print("Nb of trees: ", str(train.counter_tree_visite))
        # print("Nb of evaluated nodes: ", str(train.counter_node_visite))
        # print("Nb of evaluated feature: ", str(train.counter_feature_visite))
        # print("Nb of evaluated split: ", str(train.counter_split_feature_visite))
        # print("Nb of evaluated MoD: ", str(train.counter_MoD))
            
                
        return L_records, L_root_tree_building, L_train_pred
    
    
    
# test
#train = Train()


#L_records, L_root_tree_building, L_train_pred =  train.RF_Train(X, y, num_feat = 3, 
                                                     # n_tree = 20, sample_n = 0.8,
                                                     # min_bucket=5, max_size = 5, cores = 1,
                                                     # strategy=None, bins =None)









