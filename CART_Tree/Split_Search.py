# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:09:02 2021

@author: tyler
"""

"""
                    Split Search for Continous Variables (Not Done Adding My Comments)
"""

"""
Variable and Target To Test Function
"""

#from sklearn.datasets import load_boston
#X, Y = load_boston(return_X_y=True)
#X = X[:,6]  

"""
The goal of this class is to find the best split point given a continuous feature for a decision tree using a purity index (MSE for continuous target and Gini Index for 
categorical target used in this class). Different splitting methods used (defined in class) include:
    
    1. All Possible Splits (All_Splits): All possible splitting points are evaluated to find best split at decision tree node. This is done by looking at the adjacent values 
    of each pair of points in a given feature vector, and finding point that has lowest purity score for target variable. 
    
    2. Quantile Splits (Quantile_Split): WILL ADD
        
    3. Binning Splits (Binning_Split): WILL ADD
        
    4. Gaussian Splits (Gaussian_Split): WILL ADD

Input:
    - Feature Variable: it is assumed to be a continuous variable stored in vector
    - Target Variable: it is assumed to be a continuous variable stored in vector

User Specified Options:
    - Sample: can specify size of sample of Feature Variable observations to find potential splitting points
    - Cross Validation: NOT IMPLEMENT YET

Outputs:
    - Optimal split point based on specified splitting point method 
"""

"""
Best_Splitting_Point Class
"""

#Libraries to Import
import numpy as np
import Measure_Purity as sim

class Best_Splitting_Point(sim.MeasureOfDispersion):
    
    def __init__ (self):
        self.counter_split_feature_visite = 0
        sim.MeasureOfDispersion.__init__(self)
        
     
    
    def Splitting(self,split_points): 
        split_purity = [] #Creating empty list to append split purity values
        
        for s in split_points: #Loops over length of possible split points
        
            self.counter_split_feature_visite += 1
        
            lower = self.data[self.data[:,0]<=s,:]
            upper = self.data[self.data[:,0]>s,:]
        
            split_purity.append(self.MeasureOfDispersion(lower[:,1], upper[:,1]))
            
        split_results = np.transpose(np.array([split_points,split_purity]))
        best_purity = np.min(split_results[:,1])
        min_split = np.array(np.where(split_results[:,1] == best_purity))
        min_split = split_results[min_split[0,0],0]
        
        return min_split, best_purity
    
    #Step 4: Calling Splitting Function in Step 2 for Different Splitting Point Strategies in Step 1
    ###Note to team: Can make this into one big function where user input is split type
    
    
    def All_Points(self,feature, target, sample = None):
        self.data = np.transpose(np.array([feature,target])) 
        split_points = np.unique(self.data[:, 0]) #Find adjacent values
        optimal_split_point, best_purity = self.Splitting(split_points)
        return optimal_split_point, best_purity
 


