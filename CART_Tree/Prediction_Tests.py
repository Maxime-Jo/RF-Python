# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:46:20 2021

@author: tyler
"""

####Test####
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# X, y = load_boston(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

# #Continuous Target
#Pred = Prediction()
#y_train_pred = Pred.Predict(X_train, y_train,X_train, y_train, sample_f = 3, 
                                                 #n_tree = 10, sample_n = 0.5,
                                                   #min_bucket=5, max_size = 4)

# y_test_pred = Pred.Test_Prediction(X_train, y_train, X_test, y_test, sample_f = 3, 
#                                                  n_tree = 100, sample_n = 0.1,
#                                                    min_bucket=5, max_size = 4)


#Testing with scikit learn 
# from sklearn.ensemble import RandomForestRegressor
# regr = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0)
# regr.fit(X_train, y_train)

# y_pred_sklearn = regr.predict(X_test)

# #Accuracy
# (((y_test-y_test_pred)**2).sum())**0.5
# (((y_test-y_pred_sklearn)**2).sum())**0.5  

# #Test with binary target
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 


# y_train_pred = Pred.Train_Prediction(X, y, sample_f = 3, 
#                                                  n_tree = 10, sample_n = 0.5,
#                                                    min_bucket=5, max_size = 4)

# #Test with large target
# np.savetxt('X.csv', X, fmt='%d')
# X = np.loadtxt('X.csv', dtype=int)

# np.savetxt('y.csv', y, fmt='%d')
# y = np.loadtxt('y.csv', dtype=int)

# y_train_pred = Pred.Train_Prediction(X, y, sample_f = 3, 
#                                                  n_tree = 10, sample_n = 0.2,
#                                                    min_bucket=5, max_size = 4)
