import numpy as np
import pandas as pd
import time


class DecisionTree:
    """ Unprunded decision tree by simple C&RT algorithm using Gini index as impurity measure.
        Pruned version can be made by setting maximum height parameter.
    """
    
    # --------------------- Internal Node Class --------------------- #
    class Node:
        """Internal node class to store information in each node."""
        
        def __init__(self, height, position, branch=None, value=None):
            self.height = height
            self.position = position
            self.branch = branch
            self.value = value
    
    
    # --------------------- Binary Decision Tree Constructor --------------------- #
    def __init__(self, print_flag=False):
        """Create an empty binary tree."""
        
        self.dim = None                # dimension of the data
        self.size = 0                  # total size of the tree
        self.node_dict = {}            # internal node dictionary
        self.print_flag = print_flag
   
    
    def construct_tree(self, data, col_y, height=0, position=0, max_height=50):      
        """Construct decision tree by recursively branching each node."""
        
        self.dim = data.shape[1]
        y = col_y  # y column name
        
        # return empty tree if no data
        if (data.shape[0] == 0): return;
        
        # printing training information if set to True
        if self.print_flag:
            if position == 0: print('Start training decision tree.')
        
        # terminate when no branch can be made (all y are the same)
        if len(pd.unique(data.y)) == 1:
            value = pd.unique(data.y)[0]
            self.node_dict[position] = self.Node(height, position, value=value)
            self.size += 1
        # terminate when height reaches the maximum limit
        elif height >= max_height:
            value = 2 * (np.sum(data.y.values) >= 0) - 1
            self.node_dict[position] = self.Node(height, position, value=value)
        # recursvively branching each node
        else:
            b = self.branching(data)
            self.node_dict[position] = self.Node(height, position, b)
            best_dim, best_theta = b
            self.size += 1
            
            # Construct left sub-tree
            LeftTree = data[data[best_dim] < best_theta]
            self.construct_tree(LeftTree, y, height + 1, 2 * position + 1, max_height)
            
            if self.print_flag:
                print('  ... constructing height %d left substree at position %d' %(height+1, 2*position+1))
            
            # Construct right sub-tree
            RightTree = data[data[best_dim] >= best_theta]
            self.construct_tree(RightTree, y, height + 1, 2 * position + 2, max_height)
            
            if self.print_flag:
                print('  ... constructing height %d right substree at position %d' %(height+1, 2*position+2))
            

    def branching(self, data):
        """ Use decision stump to find the best branching criteria
            Return the best threshold of i-th feature column.
        """
        
        N = data.shape[0]

        # Learn branching criteria b(X)
        min_impurity = 1
        for i in range(self.dim - 1):
            x = np.array(data.sort_values(i)[i])
            y = np.array(data.sort_values(i).y)
            thresholds = [(x[k] + x[k + 1]) / 2 for k in range(len(x) - 1)] + [np.max(x) + 1]
            for j in range(len(thresholds)):
                theta = thresholds[j]
                y1 = y[:j + 1]
                y2 = y[j + 1:]
                weighted_impurity = (len(y1) / N) * self.gini_index(y1) + (len(y2) / N) * self.gini_index(y2)

                #print('weighted_impurity:', weighted_impurity)
                if weighted_impurity <= min_impurity:
                    best_dim, best_theta, min_impurity = i, theta, weighted_impurity

        return best_dim, best_theta
            
    
    def gini_index(self, y):
        """Gini index for impurity measure"""
        
        if (len(y) == 0): return 0
        gini = 1.0
        for k in [1.0, -1.0]:
            gini -= np.square(np.sum(y == k) / len(y))

        return gini

    
    def predict(self, x, position = 0):
        """Predict the value of one data point x, recrusively traverse each node."""
        
        node = self.node_dict[position]
        
        if node.branch:
            branch_dim, branch_theta = node.branch
            if x[branch_dim] >= branch_theta:
                return self.predict(x, position = (2 * position + 2))
            else:
                return self.predict(x, position = (2 * position + 1))
        else:
            return node.value


def generate_data(file):
    """Return data point X and lable Y."""
    data = pd.read_csv(file, sep=' ', header=None)
    X = data.loc[:, 0:1]
    Y = np.array(data.loc[:, 2:]).flatten()
    
    return X, Y


def run_model():
    
    # load data
    train_file = 'data/hw7_train.dat.txt'; test_file = 'data/hw7_test.dat.txt'
    data_train = pd.read_csv(train_file, sep = ' ', header = None, names=[0, 1, 'y'])
    data_test = pd.read_csv(test_file, sep = ' ', header = None, names=[0, 1, 'y'])
    X_train, Y_train = generate_data(train_file); X_test, Y_test = generate_data(test_file)

    # train model
    col_y = 'y'

    time_start = time.clock()
    Tree = DecisionTree(print_flag = True)
    Tree.construct_tree(data_train, col_y)

    print("Using %.3f seconds" % (time.clock() - time_start))

    # model accuracy
    print('\n--- Decision tree model accuracy ---')
    Y_train_pred = [Tree.predict(x) for x in np.array(X_train)]
    train_acc = np.sum(Y_train_pred == Y_train) / len(Y_train) * 100
    print('Accuracy on the training set: %.2f %%' %train_acc)

    Y_test_pred = [Tree.predict(x) for x in np.array(X_test)]
    test_acc = np.sum(Y_test_pred == Y_test) / len(Y_test) * 100
    print('Accuracy on the testing set: %.2f %%\n' %test_acc)


def main():
    run_model()


if __name__ == '__main__':
    main()