"""
The goal of this class is to impute missing values for the feature columns 
in a dataset. 

Methods implemented for numerical feature columns:

 - Mean:       null values in a column are filled by the mean of the column
 - Median:     null values in a column are filled by the median of the column
 - Mode:       null values in a column are filled by the mode of the column
 - knn:        null values in a column are filled by using k-Nearest Neighbors

Methods implemented for categorical feature columns:

 - Mode:       null values in a column are filled by the mode of the column
 - knn:        null values in a column are filled by using k-Nearest Neighbors

Input:
    
    impute():
    - X: matrix of features
    
    optional:
    - y: target vector (default = None, only used when knn is chosen as imputation method)
    - num_method: imputation method for numerical columns (default = mean)
    - cat_method: imputation method for categorical columns (default = mode)
    - cat_column: list of user-defined categorical column indices (only when users wish to treat numerical(binary/integer) columns as categorical)
    - k: number of neighboring samples to use for knn imputation (default = 5)
    
Outputs:
   
    - return an array of float with missing value filled by using the 
      assigned methods
    
"""

import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer

class Missing_Value:

    def __init__(self):
        self.x = None
        self.data = None
        self.cat_column = None
        self.cat_col_map = None
        self.missing_data = None
        self.missing_row = None   # missing row indices for easy referencing
        self.num_method = None
        self.cat_method = None
        self.k = None
        
        self.valid_num_method = ['mean', 'median', 'mode', 'knn']
        self.valid_cat_method = ['mode', 'knn']

    def impute(self, X, y=None, num_method='mean', cat_method='mode', cat_column=None, k=5):

        self.validate_methods(num_method, cat_method)
        
        self.k = k   
        self.num_method = num_method
        self.cat_method = cat_method
        
        self.x = np.array(X)
        self.data = self.x.transpose()     # features (in column)
        
        self.cat_column, self.cat_col_map, self.data = self.process_cat_columns(self.data)
        # combine with user-defined cat_column
        if cat_column != None: self.cat_column = list(set().union(self.cat_column, cat_column))
        
        self.data = np.array(self.data).astype(float)
        self.missing_data, self.missing_row = self.get_nan_location(self.data)
        
                        
        if num_method == 'knn': 
            self.x = self.knn(self.x, y, k)
            for i in range(len(self.cat_column)):
                col = self.cat_column[i]
                if col in self.missing_data:
                    for j in self.missing_data[col]:
                        self.x[j,col] = round(self.x[j,col])
            #print('knn')
            return np.array(self.x).astype(float)
        
        for cidx in self.missing_data:
            if cidx not in self.cat_column:      # data is numerical
                if num_method == 'mean':
                    mean = np.nanmean(self.data[cidx])
                    for ridx in self.missing_data[cidx]:
                        self.data[cidx, ridx] = mean
                    #print(cidx,'mean = ',mean)
                elif num_method == 'median':
                    median = np.nanmedian(self.data[cidx])
                    for ridx in self.missing_data[cidx]:
                        self.data[cidx, ridx] = median
                    #print(cidx,'median = ',median)
                else: # num_method == 'mode':
                    mode = stats.mode(self.data[cidx])  
                    for ridx in self.missing_data[cidx]:
                        self.data[cidx, ridx] = mode[0]
                    #print(cidx,'num mode = ',mode[0])                    

            else: # data is categorical
                # cat_method == 'mode':
                mode = stats.mode(self.data[cidx])
                for ridx in self.missing_data[cidx]:
                    self.data[cidx, ridx] = mode[0]
                #print(cidx,'cat mode = ',mode[0])                    
        return self.data.transpose()

    def validate_methods(self, num_method, cat_method):
        if num_method not in self.valid_num_method:
            raise ValueError('Invalid method for numeric features:' + num_method)
        
        if cat_method not in self.valid_cat_method:
            raise ValueError('Invalid method for categorical features:' + cat_method)
        
        # if knn is chosen, it must be used for all missing values
        if (num_method == 'knn' and cat_method != 'knn') or (num_method != 'knn' and cat_method == 'knn'):
            raise ValueError('Invalid methods: knn must be used for both numeric and categorical features')


    def process_cat_columns(self, features):
        cat_col = []
        cat_col_map = {}
        for i in range(len(features)):
            if self.isCategorical(features[i]): 
                cat_col.append(i)
                features[i], cat_col_map[i] = self.transform_cat_to_num(features[i])
        return cat_col, cat_col_map, features
            
    def isCategorical(self, column):
        isCat = False
        for j in range(len(column)):
            if type(column[j]) not in [float, int]: return True
        return isCat
    
    def transform_cat_to_num(self, column):
        c = 0
        col_map = {}
        for i in range(len(column)):
            if column[i] is not np.nan:
                if column[i] in col_map: 
                    column[i] = col_map[column[i]]
                else:
                    col_map[column[i]] = c
                    column[i] = c
                    c += 1
        return column, col_map
    
    # get the location (column,row indices) for all missing values
    def get_nan_location(self, data):
        #n = 0   # total number of columns contains missing values
        missing_loc = {}
        missing_row = []
        for i in range(len(data)):
            if np.isnan(data[i]).any():
                rlist = self.get_missing_rows(data[i])    # row indices of missing data
                missing_loc[i] = rlist
                missing_row += rlist
                #n += 1
        
        missing_row = np.unique(missing_row)
        return missing_loc, missing_row

    # get row indices of missing values
    def get_missing_rows(self, column):
        nrows = []
        for j in range(len(column)):
            if np.isnan(column[j]):
                nrows.append(j)
        return nrows
                
    def knn(self, x, y, k):
        # define imputer
        imputer = KNNImputer(n_neighbors=k, weights='distance')
        # fit to data & transform
        return imputer.fit_transform(x,y)
        
            
 
# import pandas as pd

# credit = pd.read_csv("CreditGame_TRAIN.csv")
# target_name = ["DEFAULT","PROFIT_LOSS"] # this dataset has 2 target columns

# # remove target, id and redundant columns
# X = credit.drop(columns = target_name + ["ID_TRAIN","TYP_FIN"])
# y = np.array(credit[target_name[1]])
 
# missing = Missing_Value()
# impX = missing.impute(X,y,num_method="mode", cat_method="mode")
