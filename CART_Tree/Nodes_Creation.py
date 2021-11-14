"""
Test output
"""

# Load data
#from sklearn.datasets import load_boston
#X, y = load_boston(return_X_y=True)


"""
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

"""


"""
Class Start
"""

import numpy as np
import Best_Cut as bc

feature_search = bc.Best_cut()


class NodeSearch():
    
    def stopping_criteria(self, node_level, y_records, stopping_criteria, max_size):
        
        max_node = y_records[:,-1].max()                                                    # continue until reach last node
        
        if node_level == max_node:                                                           # stopping criteria
            stopping_criteria = True
                
        if node_level > 2**max_size-1:
            stopping_criteria = True
                
        return stopping_criteria
    
    
    def next_node(self, node_level, y_records, stopping_criteria, max_size):
        
        node_level += 1  # Here next node operation
            
        stopping_criteria = self.stopping_criteria(node_level, y_records, stopping_criteria, max_size)    # stopping criteria
        
        node_level_decision = False                                                             # some nodes do not exist                                                                                                
                                                                                                # need to go to the next one
                                                                                                # un-less we reach final one
        while node_level_decision == False:  
            if sum(y_records[:,-1]==node_level) == 0:                                           # HERE, if node does not exist
                node_level += 1                                                                 # go to next one
                
                stopping_criteria = self.stopping_criteria(node_level, y_records, stopping_criteria, max_size)
                node_level_decision = stopping_criteria
              
            else : 
                node_level_decision = True                                                      # exit while loop
                        
                
        return stopping_criteria, node_level
    
    
    
    def building_constraints (self, bucket_1, bucket_2, min_bucket):                            # can we create a node?
        
        if min(len(bucket_1),len(bucket_2)) >= min_bucket:
            out = True
        else:
            out = False
            
        return out
        
    
    def create_new_node (self, cut, vector_size, y_records, node_level, min_bucket, root_tree_building, feature, cut_value):
        
        child_1 = node_level*2
        child_2 = child_1 + 1
        
        bucket_1 = [child_1]*cut
        bucket_2 = [child_2]*(vector_size-cut)
        
        if self.building_constraints(bucket_1, bucket_2, min_bucket) == True:
        
            cut_record = np.array(bucket_1 + bucket_2)                                          # create the two news branches
        
            min_index = np.squeeze(np.where(y_records[:,-1]==node_level)).min()                  # we need to re-insert the branches                                                                                                
            max_index = np.squeeze(np.where(y_records[:,-1]==node_level)).max()                  # looking at where to insert
        
                    
            left = y_records[np.squeeze(np.where(y_records[:,-1])<min_index),-1]                # get the left and the right records
            right = y_records[np.squeeze(np.where(y_records[:,-1]))>max_index,-1]               # for insertion
        
            new_record = np.concatenate((left, cut_record, right))                              # new record of the tree
        
            y_records = np.concatenate((y_records, np.transpose([new_record])), axis = 1)       # insert into the full records
            root_tree_building = np.concatenate((root_tree_building, [[node_level, cut_value, feature]]), axis = 0)
            
        return y_records, root_tree_building
    
        
    def breath_first_search(self, X, y, min_bucket = 5, max_size = 3):
        
        y_records = np.random.randint(0,1,len(y)) + 1           # initialise tree records
        y_records = np.transpose(np.array([y_records]))
        
        node_level = 1                                           # initialise tree level        
        father_X = X                                            # initialise parents
        mother_y = y                                            # initialise parents
        
        root_tree_building = np.array([[0,0,-1]])
        
        stopping_criteria = False
             
        
        while stopping_criteria == False:
            
            
            vector_size = len(mother_y)
            
            
            cut, feature, cut_value = feature_search.visit_all_features(X = father_X, y = mother_y)
            cut = cut.astype(int)
            
            y_records, root_tree_building = self.create_new_node(cut, vector_size, 
                                                                 y_records, node_level, min_bucket, 
                                                                 root_tree_building, feature, cut_value)   # node creation   
                                    
            
            stopping_criteria, node_level = self.next_node(node_level, y_records, stopping_criteria, max_size) # go to next node
                
            
            father_X = X[y_records[:,-1]==node_level,:]                                              # update the new parents    
            mother_y = y[y_records[:,-1]==node_level]
            
            print(node_level)
            
        return y_records, root_tree_building
            
    

"""
Test output
"""
#NS = NodeSearch()

#test = NS.breath_first_search(X, y)


#y_records = test

#y_records, root_tree_building = NS.breath_first_search(X_train,y_train)





















