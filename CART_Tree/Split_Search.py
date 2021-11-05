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
import statistics as stats
import scipy.stats
import Measure_Purity as sim

class Best_Splitting_Point:
     
    #Step 0: Initializing split variable (feature) and target variable (target)
    def __init__(self, feature, target):                            #Initialization line that takes in feature and target variable
        self.data = np.transpose(np.array([feature,target]))        #Creating 2 dimensional array for feature and target variable
        self.data = self.data[self.data[:, 0].argsort()]            #Sorting Feature variables (ascending)
    
    #Step 1: Finding Potential Splitting Points using Different Methods
    
    #A. Adjacent Values (All Possible Splits)
    def Adjacent_Values(self,values,sample):
        adjacent_list = []                                          #Creating list to store adjacent values
        final_values =  np.array(self.Sampling(values, sample))     #Sampling values (if sample call used in Step 4)
        for i in range(1, len(final_values)):                       #Finding and storing all adjacent values for feature variable
            adjacent = (final_values[i-1]+final_values[i])/2
            adjacent_list.append(adjacent)
        adjacent_list = np.unique(adjacent_list)
        return adjacent_list
        
    #B. Quantile Values (Specify Number of Quantiles to Use)
    def Quantile_Values(self,values, quantiles, sample):
        quantile_list = []                                          #Creating list to store quantile values
        final_values =  np.array(self.Sampling(values, sample))     
        for i in range(0, quantiles):                               #Finding and storing all quantiles values based on number of quantiles specified
            quant_input = i/(quantiles-1)
            quant_value = np.quantile(final_values, quant_input)
            quantile_list.append(quant_value)
        return quantile_list
    
    #C. Binning Values (Specify Number of Bins to Use) #GET SOURCE
    def Binning_Values(self,values, bins, sample):
        final_values =  np.array(self.Sampling(values, sample))
        bins_values = np.linspace(min(final_values), max(final_values), bins)
        digitized = np.digitize(final_values, bins_values)
        binning_list = list([final_values[digitized == i].mean() for i in range(1, len(bins_values))])     
        return binning_list
    
    #D. Gaussian Split Values (Specify Number of Splits to Use)
    def Gaussian_Values(self, values, splits, sample):
        gaussian_list = []
        final_values =  np.array(self.Sampling(values, sample))
        for i in range(1, splits+1):
            gaussian_input = i/(splits+2)
            norm_zvalue = scipy.stats.norm.ppf(gaussian_input)
            norm_value = norm_zvalue*stats.stdev(final_values)+ stats.mean(final_values)
            gaussian_list.append(norm_value)
        return gaussian_list
    
    #Step 2: Defining Sampling Procedure for Sample Call in Step 4
    def Sampling(self, values, sample_size):
        sampled_set = []
        random_rows = np.random.randint(len(values), size = sample_size)
        for i in random_rows:
            sample = values[i]
            sampled_set.append(sample)
        return sampled_set
    
    def Empty_Sample(self, sample):
        if sample is None:
            sample = len(self.data[:, 0])
        else:
            sample = sample
        return sample
    
    #Step 3: Splitting Feature Based on Specified Strategy
    #Returns Best Split Point Based on Error Measures Defined in 'GINIRMSE'
    
    def Splitting(self,split_points): 
        split_purity = [] #Creating empty list to append split purity values
        
        for i in range(0, len(split_points)): #Loops over length of possible split points
            lower_x = []
            upper_x = []
            lower_target = []
            upper_target = []
            
            for h in range(0,len(self.data)-1):
                if self.data[h,0] <= split_points[i]:
                    lower_x.append(self.data[h,0])
                    lower_target.append(self.data[h,1])
                else:   
                    upper_x.append(self.data[h,0])
                    upper_target.append(self.data[h,1])
    
            lower =  np.transpose(np.array([lower_x,lower_target]))
            upper =  np.transpose(np.array([upper_x,upper_target]))
        
            MoD = sim.MeasureOfDispersion()
            split_purity.append(MoD.MeasureOfDispersion(lower[:,1], upper[:,1]))
            
        split_results = np.transpose(np.array([split_points,split_purity]))
        best_purity = np.min(split_results[:,1])
        min_split = np.array(np.where(split_results[:,1] == best_purity))
        min_split = split_results[min_split[0,0],0]
        
        return min_split, best_purity
    
    #Step 4: Calling Splitting Function in Step 2 for Different Splitting Point Strategies in Step 1
    ###Note to team: Can make this into one big function where user input is split type
    
    #A. All Splits
    def All_Splits(self, sample = None):
        sample = self.Empty_Sample(sample)
        adjacent_split_points = self.Adjacent_Values(self.data[:, 0], sample) #Find adjacent values
        optimal_split_point, best_purity = self.Splitting(adjacent_split_points)
        return optimal_split_point, best_purity
    
    #B. Quantile Split (Specify Number of Quantiles to Use)
    def Quantile_Split(self, quantiles, sample = None): #User Can specify number of quantiles
        sample = self.Empty_Sample(sample)
        quantile_split_points = self.Quantile_Values(self.data[:, 0],quantiles, sample) #Find Quantile values
        optimal_split_point, best_purity = self.Splitting(quantile_split_points)
        return optimal_split_point, best_purity      
    
    #C. Binning Split (Specify Number of Bins to Use)
    def Binning_Split(self, bins, sample = None): #User Can specify number of quantiles
        sample = self.Empty_Sample(sample)
        binning_split_points = self.Binning_Values(self.data[:, 0],bins, sample) #Find Quantile values
        optimal_split_point, best_purity = self.Splitting(binning_split_points)
        return optimal_split_point, best_purity

    #D. Gaussian Split (Specify Number of Splits to Use)
    def Gaussian_Split(self, splits, sample = None): #User Can specify number of quantiles
        sample = self.Empty_Sample(sample)
        gaussian_split_points = self.Gaussian_Values(self.data[:, 0],splits, sample) #Find Quantile values
        optimal_split_point, best_purity = self.Splitting(gaussian_split_points)
        return optimal_split_point, best_purity   

"""
Example using Four Different Methods
"""        

#Calling defined class
#Tree_Split = Best_Splitting_Point(X,Y)

#A. Sort Split
#All_Split_Point = Tree_Split.All_Splits(sample = 100)

#B. Quantile Split
#Quantile_Split_Point = Tree_Split.Quantile_Split(quantiles = 10, sample = 100)

#C. Binning Split
#Binning_Split_Point = Tree_Split.Binning_Split(bins = 10, sample = 100)

#D. Gaussian Split
#Gaussian_Split_Point = Tree_Split.Gaussian_Split(splits = 10, sample = 100)


"""Other Ideas to Implement:
    1. Cross valiation to select splitting point
    2. Other shortcuts to simplify number of points to visit
"""
    
