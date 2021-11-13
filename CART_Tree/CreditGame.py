import pandas as pd
import Missing_Value as MV
import Transform_Categorical as TC
import Measure_Purity as MP
import Split_Search as SS


# load dataset
credit = pd.read_csv("CreditGame_TRAIN.csv")

target_name = ["DEFAULT","PROFIT_LOSS"] # this dataset has 2 target columns
data = credit.drop(columns=target_name)

# Pre-processing

# 'TYP_FIN' column is not informative, only contains one value: 'AUTO'
data = data.drop(columns=["ID_TRAIN","TYP_FIN"])

# mapping of categorical variables
data['TYP_RES'] = data['TYP_RES'].map({'L':0, 'A':1, 'P':2})
data['ST_EMPL'] = data['ST_EMPL'].map({'R':0, 'T':1, 'P':2})

arr = data.to_numpy()
impX = MV.Missing_Value(arr)
impX.impute(num_method="mean", cat_method="mode")

X = impX.data.transpose()
y = credit[target_name[0]].to_numpy()

# Transform_Categorical if necessary

Tree_Split = SS.Best_Splitting_Point(X,y)
Gaussian_Split_Point = Tree_Split.Gaussian_Split(splits = 10, sample = 100)
