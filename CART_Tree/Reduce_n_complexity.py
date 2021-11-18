#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:37:50 2021

@author: maxime
"""

# Load data
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)


import numpy as np

class Reduce_Complexity:
    
    def binning(self, x, bins):
        x_unique = np.unique(x)
        x_max = max(x_unique)
        x_min = min(x_unique)
        step = (x_max-x_min)/bins
        x = np.floor_divide(x,step)*step
        return x
        
    def quantile_binning(self, x, bins):
        x_unique = np.unique(x, return_counts = True)[0]
        x_count = np.unique(x, return_counts = True)[1]
        count_sum = x_count.sum()
        step = count_sum / bins
        x_count_cumsum = np.cumsum(x_count)
        x_count_bins =  np.floor_divide(x_count_cumsum,step)
        
        for i in range(0,len(x_unique)):
            x[x==x_unique[i]] = x_unique[x_count_bins==x_count_bins[i]].max()
            
        return x
            
    def reduce(self,X,bins,strategy):
        
        X_trans = np.copy(X)
        
        for f in range(0,X.shape[1]):
            if strategy == "bin":
                X_trans[:,f] = self.binning(X[:,f],bins)
            elif strategy == "quant":
                X_trans[:,f] = self.quantile_binning(X[:,f],bins)
            else:
                X_trans = X
        return X_trans
                
#RC = Reduce_Complexity()

#X_trans = RC.reduce(X,5,"quant")
    
