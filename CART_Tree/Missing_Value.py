"""
The goal of this class is to impute missing values for the feature columns 
in a dataset. 

Methods implemented for numerical feature columns:

 - Mean:       null values in a column are filled by the mean of the column
 - Median:     null values in a column are filled by the median of the column
 - Mode:       null values in a column are filled by the mode of the column
 - Regression: null values in a column are filled by fitting a linear 
               regression model using other columns in the dataset.

Methods implemented for categorical feature columns:

 - Mode:       null values in a column are filled by the mode of the column
 - Regression: null values in a column are filled by fitting a logistic 
               regression model using other columns in the dataset.

    
Input:
    
    __init__():
    - matrix that contains column vectors of both features and response 
      (2d numpy array)
    - index/indices of the categorical column/columns (list of integers)
      (default is an empty list)

    impute():
    - imputation method for numerical columns
    - imputation method for categorical columns (default = mode)
    
Outputs:
    
    impute():
    - return the input matrix with missing value filled by using the 
      assigned methods
    
"""

import numpy as np
import statistics as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

class Missing_Value:

    def __init__(self, columns, cat_col=[]):
        self.data = np.array(columns)
        self.cat_column = cat_col
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
    
    # creates dummy variables for categorical variables
    def __create_dummies(self, target_column):
        import pandas as pd
        dummy = pd.get_dummies(self.data[target_column], 
                               drop_first=True).to_numpy()
        return(dummy)
    
    # target_column: column index of the missing values
    # data_type: 1 for quantitative and 0 for categorical
    def __regression(self, target_column, is_numeric):
        all_data = np.transpose(self.data)
        curr_target = target_column
        for cat_col in self.cat_column:
            if cat_col != target_column:
                dum = self.__create_dummies(cat_col)
                # delete original categorical column and add dummies
                all_data = np.delete(np.concatenate((all_data, dum), axis=1), 
                                     cat_col, axis=1)
                if target_column > cat_col: curr_target -= 1 
        
        train_set = np.delete(all_data, self.missing_row,axis=0)    
        test_set = all_data[self.missing_data[target_column][1]]

        X_train = np.delete(train_set, [curr_target], axis=1)
        Y_train = train_set[:,curr_target]
        X_test = np.delete(test_set,[curr_target], axis=1)
        
        if is_numeric:                      # target data is numerical
            model = LinearRegression()
        else: 
            #model = LogisticRegression()    # target data is categorical
            model = LogisticRegression(max_iter=4000)
                
        model.fit(X_train,Y_train)
        Y_test = model.predict(X_test)
        return np.transpose(Y_test)

    def impute(self, num_method, cat_method='mode'):
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
                elif num_method == "regr":
                    regr = self.__regression(cidx, self.missing_data[cidx][0])
                    self.data[cidx, self.missing_data[cidx][1]] = regr
                else: raise ValueError('Invalid method') 
            else:                               # data is categorical
                if cat_method == 'mode':
                    mode = self.__get_mode(cidx)    
                    for ridx in self.missing_data[cidx][1]:
                        self.data[cidx, ridx] = mode
                elif cat_method == "regr":
                    regr = self.__regression(cidx, self.missing_data[cidx][0])
                    self.data[cidx, self.missing_data[cidx][1]] = regr
                else: raise ValueError('Invalid method') 
    

# test
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

# to create some missing values
X_nan = X.copy()
X_nan[150:165,3] = np.nan     # orig: [0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0.]
X_nan[26:30,6] = np.nan       # orig: [90.3, 88.8, 94.4, 87.3]
X_nan[251:260,8] = np.nan     # orig: [7., 7., 7., 1., 1., 3., 5., 5., 5.]

c = np.column_stack((X_nan,y)).transpose()

#missing = Missing_Value(c)
missing = Missing_Value(c, cat_col=[3,8])
missing.impute(num_method="regr", cat_method="regr")
X_imputed = missing.data[0:len(missing.data)-1,:].transpose()

from sklearn.metrics import accuracy_score, mean_squared_error

print(X[150:165,3])
print(X_imputed[150:165,3])
print("{:10.3f}".format(accuracy_score(X[150:165,3], X_imputed[150:165,3])))

print(X[26:30,6])
print(X_imputed[26:30,6])
print("{:10.3f}".format(mean_squared_error(X[26:30,6], X_imputed[26:30,6])))

print(X[251:260,8])
print(X_imputed[251:260,8])
print("{:10.3f}".format(accuracy_score(X[251:260,8], X_imputed[251:260,8])))


#************************************************
# test for KNN

from numpy import isnan
from sklearn.impute import KNNImputer

# print total missing
print('Missing: %d' % sum(isnan(X_nan).flatten()))

# define imputer
imputer = KNNImputer(n_neighbors=3, weights='distance', metric='nan_euclidean')
# fit on the dataset
imputer.fit(X_nan)
# transform the dataset
Xtrans = imputer.transform(X_nan)
# print total missing
print('Missing: %d' % sum(isnan(Xtrans).flatten()))

print(X[150:165,3])
print(Xtrans[150:165,3])
print("{:10.3f}".format(accuracy_score(X[150:165,3], np.round(Xtrans[150:165,3]))))

print(X[26:30,6])
print(Xtrans[26:30,6])
print("{:10.3f}".format(mean_squared_error(X[26:30,6], Xtrans[26:30,6])))

print(X[251:260,8])
print(Xtrans[251:260,8])
print("{:10.3f}".format(accuracy_score(X[251:260,8], np.round(Xtrans[251:260,8]))))