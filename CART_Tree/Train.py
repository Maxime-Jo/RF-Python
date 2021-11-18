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
    
    
    def CART_Train(self,X, y, sample_f = None, 
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
                                                               max_size = max_size, sample_f=sample_f)
        
        return y_records, root_tree_building
    
    
    def RF_Train(self, X, y, sample_f = 3, 
                 n_tree = 1, sample_n = None,
                 min_bucket=1, max_size = 6, cores = 1, 
                 strategy=None, bins = None):
        
        L_records = []
        L_root_tree_building = []
        
        cores = min(cores, multiprocessing.cpu_count()-1)
        
        if cores == 1:     
            for n in range(0,n_tree):
                    print("#######################")
                    print("TREE: "+str(n))
                    print("#######################")
                    
                    
                    """ Get tree """
                    y_records, root_tree_building = self.CART_Train(X, y, sample_f = sample_f, 
                                                                    sample_n = sample_n,
                                                                    min_bucket=min_bucket, max_size = max_size,
                                                                    strategy=strategy, bins = bins)
            
                    """ Append results """
                    L_records.append(y_records)
                    L_root_tree_building.append(root_tree_building)
                    
            
        else:            
            processed_list = Parallel(n_jobs=cores)(delayed(self.CART_Train)(X, y, sample_f = sample_f, 
                                                                    sample_n = sample_n,
                                                                    min_bucket=min_bucket, max_size = max_size,
                                                                    strategy=strategy, bins = bins) for i in range(0,n_tree))
            
            for tree in processed_list:
                y_records, root_tree_building = tree
    
                """ Append results """
                L_records.append(y_records)
                L_root_tree_building.append(root_tree_building)
                
                
        print("#######################")
        print("TRAINING DONE")
        print("#######################")
        
        print("Nb of trees: ", str(train.counter_tree_visite))
        print("Nb of evaluated nodes: ", str(train.counter_node_visite))
        print("Nb of evaluated feature: ", str(train.counter_feature_visite))
        print("Nb of evaluated split: ", str(train.counter_split_feature_visite))
        print("Nb of evaluated MoD: ", str(train.counter_MoD))
            
                
        return L_records, L_root_tree_building
    
    
    
# test
# train = Train()


# L_records, L_root_tree_building =  train.RF_Train(X, y, sample_f = 3, 
#                                                     n_tree = 100, sample_n = 0.8,
#                                                     min_bucket=5, max_size = 5, cores = 1,
#                                                     strategy=None, bins =None)









