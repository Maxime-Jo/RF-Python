# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:09:02 2021

@author: tyler
"""

"""
                    Split Search for Continous Variables (Not Done Adding My Comments)
"""

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

# class Best_Splitting_Point(sim.MeasureOfDispersion):
    
#     def __init__ (self):
#         self.counter_split_feature_visite = 0
#         sim.MeasureOfDispersion.__init__(self)
        
#     #Step 1: Splitting Function
    
#     def Splitting(self): 

#         best_purity = float('inf')
        
#         for s in self.split_points: #Loops over length of possible split points
        
#             self.counter_split_feature_visite += 1
        
#             lower = self.y[self.x<=s]
#             upper = self.y[self.x>s]
            
#             purity = self.MeasureOfDispersion(lower, upper)
            
#             if purity < best_purity:
#                 best_purity = purity
#                 min_split = s
        
#         return min_split, best_purity
    
#     #Step 2: Calling Splitting Function in Step 1 for Different Splitting Point Strategies in Step 1
#     ###Note to team: Can make this into one big function where user input is split type
    
    
#     def All_Points(self,x, y):
#         self.x = x
#         self.y = y
#         self.split_points = np.unique(self.x) #Find adjacent values
#         optimal_split_point, best_purity = self.Splitting()
#         return optimal_split_point, best_purity
 






#BSP = Best_Splitting_Point()
#BSP.All_Points(x1, y1)
            
    
class Best_Splitting_Point(sim.MeasureOfDispersion):
    
    def __init__ (self):
        self.counter_split_feature_visite = 0
        sim.MeasureOfDispersion.__init__(self)
        
    #Step 1: Splitting Function
    
    def Splitting(self): 

        best_purity = float('inf')
        
        x = self.x
        y = self.y
        
        x_sorted, x_unique_id = np.unique(x,return_index=False, return_inverse=True)
        y_sorted = np.bincount(x_unique_id, weights=y)
        n_sorted = np.bincount(x_unique_id, weights=np.ones(y.shape))
        
        n = len(y)
        n_agg = len(y_sorted)
        y_sum = sum(y)
        y_squared_sum = sum(y**2)
        running_sum = 0
        running_sum_squared = 0
        counter = 0
        
        if n_agg == 1:
            best_purity = 0
            min_split = x_sorted[0]
        
        for i in range(0,n_agg-1):
            
            counter += n_sorted[i]
            running_sum += y_sorted[i] 
            running_sum_squared += (y_sorted[i])**2
                                     
            if (y.dtype == 'bool'):
                gini_1 = self.Gini_Score(counter, running_sum)
                gini_2 = self.Gini_Score(n-counter, y_sum-running_sum)                
                purity = self.Weighted_Avg(counter,gini_1,(n-counter),gini_2)                
            else:
                SSE_1 = self.SSE_Score(running_sum_squared, running_sum, counter)
                SSE_2 = self.SSE_Score((y_squared_sum - running_sum_squared), (y_sum - running_sum), (n-counter))
                purity = SSE_1 + SSE_2
            
            if purity < best_purity:
                best_purity = purity
                min_split = x_sorted[i]
        
        return min_split, best_purity
    
    #Step 2: Calling Splitting Function in Step 1 for Different Splitting Point Strategies in Step 1
    ###Note to team: Can make this into one big function where user input is split type
    
    
    def All_Points(self,x, y):
        self.x = x
        self.y = y
        optimal_split_point, best_purity = self.Splitting()
        return optimal_split_point, best_purity






