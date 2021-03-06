# Summary
The goal of this project is to reproduce the Random Forest algorithm from scratch and with the lowest usage of external libraries.

We will propose to the user as much as flexibility as possible:
- Categorical vs Regression
- Categorical feature transformation (One-Hot-Encoding, Target-Encoding, Frequency-Encoding)
- Possibility to handle missing values with different sort of imputation
- Parallelisation of the CART tree building
- etc.

Building the random forest from scratch means:
- Building the CART tree algorithm
- Bagging of the data
- Random sampling of the feature

Based on our reading on CATBoost, XGBoost and LightGBM, we will introduce ideas to faster and improve our algorithm:
- Depth Search vs Breath Search
- Binning of the feature
- Categorical to Continuous strategies
- Cross validation between split search and node evaluation
- etc.

...

# Members
In alphabetique order:
- Lin Lin
- Maxime Jousset
- Tyler Schwartz

# Chosen problem


# Methodology
azerty

# Process

## Description of the CART Tree
<img width="1760" alt="Screenshot 2021-09-24 at 16 05 57" src="https://user-images.githubusercontent.com/78235958/134733667-64c16d89-53a0-4813-9795-5237c82ee11a.png">

### Categorical Variable Encoding
The goal of this class is to transform any categorical vector into either one-hot-encoding matrix either in a continuous vector.

#### One-Hot-Encoding:
If c represents the number of different classes, the one-hot-encoding is build c-1 columns for these classes. The first class is ignored.
    
#### Target Encoding:
The idea of target encoding has been used by CATBoost a famous gradient boosting algorithm specialised for categorical variables. It is proposed as one-hot-encoding my consum too much memory when the number of category is large. Additionnally, one-hot-encoding does not encode any specific information regarding the category unlike embeddings (Word2Vec for the most famous in NLP).

The goal of target encoding is to use the target information to encode the categories. The value of the category will then be equal to the mean of response for this category. We have added a prior to the mean in order to account for low frequency of certain category. The prior account for 30 elements but can be changed by the user. The prior will consider the overall mean of the response.
    
#### Frequency Encoding:
The idea is similar to that of the target encoding. The frequency of the category is used instead of the response mean.
    
#### Input:
- any vector: it is assumed to be a categorical variable
- the format needs to be numpy array
    
#### Output:
- one-hot-encoding: a numpy array containing c-1 columns where c represents the number of categories
- target encoding or frequency encoding:
	- a continuous vector with the new categories embedding
	- a reference matrix to mappe the categories later (e.g., test data set)
    
### Gini Index and MSE
The goal of this class is to compute either the GINI index in case of classification or MSE in case of regression.

The class receinve as input either one or two vectors:
- If the vector(s) are boolean, it will compute the weighted average of the Gini Index.
- If the vector(s) are not boolean (float or int), it will compute (the sum of) the sum of squared errors.
    
#### Input:
- vector y1
- vector y2 - **optional**, but need to be provided as empty numpy.array: np.array([])
    
#### Output:
- gini index if y1 (and y2) are boolean
- MSE otherwise

### Node construction
#### Breath First Search (i.e., symmetric trees)
In this script, we are looking at the tree construction. This script is highly dependent on the best split search. The inputs are
the cut function/class, the covariates and the response.

The approach is following the idea of the HeapTree - especially, the fact that a node has two children which are n*2 and n2+1.

The class will visit each node one after the other. The strategy is then a breath first search approach.

We have as output the tree's construction sequence (i.e, each time we are adding a new layer in the matrix that allocate the observations
to a final leave). Here, we need to insert in the new sequence the id for the two new nodes only.

Additionnally, since we have building constraints (e.g., min leaf size), we might skip the construction of a node.

We stop the construction of the tree when we have visited all nodes.

Sometime, the next node does not exist since a constraint blocked its construction. We need to intorduce a while loop in order to
visit the next existing one.

Input:
covariates, responses, min_bucket constraints, !!! best split !!!
    
Output:
matrix that presents the tree building process

# Problems Encountered
azerty

# Results
azerty

# Bibliography
Target Encoding: https://maxhalford.github.io/blog/target-encoding/
CATBoost: https://arxiv.org/abs/1810.11363
