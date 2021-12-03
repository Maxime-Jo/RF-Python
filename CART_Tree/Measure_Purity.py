"""
Variable to test the class
"""
# Load data
#import numpy as np

#y_1 = np.array(np.random.randint(2, size=100), dtype = 'bool')
#y_2 = np.array(np.random.randint(2, size=1000), dtype = 'bool')

#y_3 = np.random.random(size=100)*100
#y_4 = np.random.random(size=1000)*100

"""
The goal of this class is to compute either the GINI index in case of classification or MSE in case of regression.

The class receinve as input either one or two vectors:
    - If the vector(s) are boolean, it will compute the weighted average of the Gini Index.
    - If the vector(s) are not boolean (float or int), it will compute (the sum of) the sum of squared errors.
    
Input:
    - vector y1
    - vector y2 - optinal but need to be provided as empty numpy.array: np.array([])
    
Output:
    - gini index if y1 (and y2) are boolean
    - MSE otherwise
"""



"""
The Class itself
"""

class MeasureOfDispersion:
    
    def __init__ (self):
        self.counter_MoD = 0

    
    def Gini_Index (self, y):
        
        n = len(y)
        bool_1 = sum(y)
        bool_0 = n - bool_1
        gini = 1 - (bool_0/n)**2 - (bool_1/n)**2
        
        return gini
    
    def Gini_Score (self, n, ones):
        
        bool_1 = ones
        bool_0 = n - bool_1
        gini = 1 - (bool_0/n)**2 - (bool_1/n)**2
        
        return gini
    
    def SSE_Score (self, y_squared_sum, y_sum, n):
        
        var = (y_squared_sum/n) - (y_sum/n)**2
        sse = var*n
        
        return sse
    
    def MSE (self, y):
        
        mean = y.mean()
        
        mse = ((y-mean)**2).sum()
        
        return mse
    
    def Weighted_Avg (self, n1, v1, n2, v2):
        
        wgt_avg = (n1*v1 + n2*v2)/(n1+n2)
        
        return wgt_avg
    
    def MeasureOfDispersion (self, y1, y2):
        
        self.counter_MoD += 1
        
        if len(y2) == 0:                                            # in case we want to calculate for only one vector
            if y1.dtype == 'bool'  :              
                out = self.Gini_Index(y1)                           # gini index
            else:
                out = self.MSE(y1)                                  # Mean Sqaured Error
                
        else:
            if (y1.dtype == 'bool') & (y2.dtype == 'bool'):         # in case we have two vectors
                n1 = len(y1)
                gini1 = self.Gini_Index(y1)
                n2 = len(y2)
                gini2 = self.Gini_Index(y2)
                
                out = self.Weighted_Avg(n1, gini1, n2, gini2)       # Weighted average of the Gini Index
                
            else:
                out = self.MSE(y1) + self.MSE(y2)                   # Sum of the mean sqared error
                
        return out

"""
Test the class output
"""

    
# MoD = MeasureOfDispersion()

# test_Gini_1var = MoD.MeasureOfDispersion(y_1, np.array([]))         # Example only one vector - GINI
# test_Gini_2var = MoD.MeasureOfDispersion(y_1, y_2)                  # Example two vectors - GINI

# test_MSE_1var = MoD.MeasureOfDispersion(y_3, np.array([]))          # Example only one vector - MSE
# test_MSE_2var = MoD.MeasureOfDispersion(y_3, y_4)                   # Example two vectors - MSE
