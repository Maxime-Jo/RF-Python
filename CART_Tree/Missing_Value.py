"""
The goal of this class is to impute missing values for the feature columns 
in a dataset. 

Methods implemented for numerical feature columns:

 - Mean:       null values in a column are filled by the mean of the column
 - Median:     null values in a column are filled by the median of the column
 - Mode:       null values in a column are filled by the mode of the column
 - knn:        null values in a column are filled by using (the mean of ) 
               k-Nearest Neighbors

Methods implemented for categorical feature columns:

 - Mode:       null values in a column are filled by the mode of the column
 - knn:        null values in a column are filled by using (the mean of ) 
               k-Nearest Neighbors

    
Input:
    
    __init__():
    - matrix of X (features)

    impute():
    - imputation method for numerical columns
    - imputation method for categorical columns (default = mode)
    - k (optional, default = 5, for kNN imputation only)
    
Outputs:
    
    impute():
    - return the input matrix with missing value filled by using the 
      assigned methods
    
"""

import numpy as np
import statistics as stats
from sklearn.impute import KNNImputer
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression

class Missing_Value:

    def __init__(self, X):
        self.X = X
        columns = X.transpose()  # column of features
        self.data = np.array(columns)
        self.cat_column = []  ### TO-DO: find categorical columns (is it necessary?)
        self.missing_data = {}
        self.missing_row = []   # missing row indices for easy referencing
        #self.missing_col_isNum = []
        
        for i in range(0, len(columns)):
            if np.isnan(columns[i]).any():
                is_num, rlist = self.__missing_rows(i)    # row indices of missing data
                self.missing_data[i] = is_num, rlist
                self.missing_row += rlist
                #self.missing_col_isNum.append([i,is_num])
        self.missing_row = np.unique(self.missing_row)
    
    
    # get row indices of missing values
    # and check if the column contains numerical data
    def __missing_rows(self, column_idx):
        col = self.data[column_idx]
        nrows = []
        isNum = True
        for i in range(0, len(col)):
            if np.isnan(col[i]):
                nrows.append(i)
            else: isNum = (isNum and self.__isNumeric(column_idx, col[i]))
        return isNum,nrows
    
    # Return: True if data is numerical
    #         False if data is categorical
    def __isNumeric(self, column_idx, data):
        if column_idx in self.cat_column:
            return False
        else:
            if type(data.item()) in [float, int]: return True 
        
            print("Data type: " + type(data.item()) + 
                  "; column " + str(column_idx) + " contains categorical data")
            self.cat_column.append(column_idx)
            return False
        
    def __get_mean(self, col_idx):
        notNA = np.delete(self.data[col_idx], self.missing_data[col_idx][1])
        return stats.mean(notNA)
       
    def __get_median(self, col_idx):
        notNA = np.delete(self.data[col_idx], self.missing_data[col_idx][1])
        return stats.median(notNA)
    
    def __get_mode(self, col_idx):
        notNA = np.delete(self.data[col_idx], self.missing_data[col_idx][1])
        # for simplicity return first mode in case of multiple modes
        return stats.mode(notNA)

    def __knn(self, k):
        print("in knn")
        # define imputer
        imputer = KNNImputer(n_neighbors=k, weights='distance', metric='nan_euclidean')
        # fit on the dataset
        imputer.fit(self.X)
        # transform the dataset
        self.X = imputer.transform(self.X)

    def impute(self, num_method, cat_method='mode', k=5):
        ### TO-DO: check the validity of arguments - value for K & raise ValueError if necessary
                
        for cidx in self.missing_data:
            if self.missing_data[cidx][0]:      # data is numerical
                if num_method == 'mean':
                    mean = self.__get_mean(cidx)
                    for ridx in self.missing_data[cidx][1]:
                        self.data[cidx, ridx] = mean
                elif num_method == 'median':
                    median = self.__get_median(cidx)
                    for ridx in self.missing_data[cidx][1]:
                        self.data[cidx, ridx] = median
                elif num_method == 'mode':
                    mode = self.__get_mode(cidx)    
                    for ridx in self.missing_data[cidx][1]:
                        self.data[cidx, ridx] = mode
                elif num_method == "knn":
                    print(str(cidx) + ": numerical")
                    self.__knn(k)
                else: raise ValueError('Invalid method') 
                
                if num_method != "knn": self.X = self.data.transpose()
            else:                               # data is categorical
                if cat_method == 'mode':
                    mode = self.__get_mode(cidx)    
                    for ridx in self.missing_data[cidx][1]:
                        self.data[cidx, ridx] = mode
                    self.X = self.data.transpose()
                elif cat_method == "knn":
                    print(str(cidx) + ": categorical")
                    self.__knn(k)
                    for ridx in self.missing_data[cidx][1]:
                        print(str(ridx)+": before = "+str(self.X[cidx, ridx]))
                        self.X[cidx, ridx] = np.round(self.X[cidx, ridx])
                        print(str(ridx)+": after  = "+str(self.X[cidx, ridx]))
                    
                else: raise ValueError('Invalid method')

            
 #%%

# # test
# from sklearn.datasets import load_boston
# X, y = load_boston(return_X_y=True)

# # to create some missing values
# X_nan = X.copy()
# X_nan[150:165,3] = np.nan     # orig: [0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0.]
# X_nan[26:30,6] = np.nan       # orig: [90.3, 88.8, 94.4, 87.3]
# X_nan[251:260,8] = np.nan     # orig: [7., 7., 7., 1., 1., 3., 5., 5., 5.]

# #c = np.column_stack((X_nan,y)).transpose()

# missing = Missing_Value(X_nan)
# missing.impute(num_method="knn", cat_method="knn", k=3)
# #X_imputed = missing.data[0:len(missing.data)-1,:].transpose()
# X_imputed = missing.X

# from sklearn.metrics import accuracy_score, mean_squared_error

# print(X[150:165,3])
# print(X_imputed[150:165,3])
# print("{:10.3f}".format(accuracy_score(X[150:165,3], X_imputed[150:165,3])))

# print(X[26:30,6])
# print(X_imputed[26:30,6])
# print("{:10.3f}".format(mean_squared_error(X[26:30,6], X_imputed[26:30,6])))

# print(X[251:260,8])
# print(X_imputed[251:260,8])
# print("{:10.3f}".format(accuracy_score(X[251:260,8], X_imputed[251:260,8])))

# #%%
# #************************************************
# # test for KNN

# # print total missing
# print('Missing: %d' % sum(np.isnan(X_nan).flatten()))

# # define imputer
# imputer = KNNImputer(n_neighbors=3, weights='distance', metric='nan_euclidean')
# # fit on the dataset
# imputer.fit(X_nan)
# # transform the dataset
# Xtrans = imputer.transform(X_nan)
# # print total missing
# print('Missing: %d' % sum(np.isnan(Xtrans).flatten()))

# print(X[150:165,3])
# print(Xtrans[150:165,3])
# print("{:10.3f}".format(accuracy_score(X[150:165,3], np.round(Xtrans[150:165,3]))))

# print(X[26:30,6])
# print(Xtrans[26:30,6])
# print("{:10.3f}".format(mean_squared_error(X[26:30,6], Xtrans[26:30,6])))

# print(X[251:260,8])
# print(Xtrans[251:260,8])
# print("{:10.3f}".format(accuracy_score(X[251:260,8], np.round(Xtrans[251:260,8]))))

