import numpy as np
import pandas as pd
import time
from DecisionTree import DecisionTree


class GradientBoostedDecisionTree:
    """Gradient boosted decision tree using pruned C&RT as base learner."""
    
    def __init__(self):
        self.trees = {}     # internal tree dictionary
        self.coeff = {}     # internal regression coefficient
    
    
    def train(self, data, col_y, iteration=50, max_height=2, print_flag=False):
        s = np.zeros((data.shape[0]))    # initialize s vector
        self.max_height = max_height
        X = data.drop(col_y, axis = 1)
        y = data[col_y]
        
        if print_flag:
            print('Start training')
            
        for i in range(iteration):
            if print_flag:
                if i % 10 == 9:
                    print(' ... %i-th iteration' %(i+1))
            
            data[col_y] = y - s
            Tree = DecisionTree()
            Tree.construct_tree(data, col_y, max_height=self.max_height)
            self.trees[i] = Tree
            
            # One variable linear regression (fit residual)
            g_t = np.array([Tree.predict(x) for x in np.array(X)])   # prediction
            if np.sum(g_t ** 2) == 0:
                alpha = 0
            else:    
                alpha = np.sum(g_t * (y - s)) / np.sum(g_t ** 2)     # compute regression coefficient
            self.coeff[i] = alpha
            s += alpha * g_t                                         # update s
        
        if print_flag:
            print('-----  END  -----')
            
    
    def predict(self, X):
        prediction = np.zeros((X.shape[0]))
        
        for i in range(len(self.trees)):
            g_t = np.array([self.trees[i].predict(x) for x in np.array(X)])
            prediction += self.coeff[i] * g_t
        
        return np.sign(prediction)


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
	time_start = time.clock()
	GBDTree = GradientBoostedDecisionTree()
	GBDTree.train(data_train, 'y', print_flag=True)
	print("Using %.3f seconds" % (time.clock() - time_start))

	# model accuracy
	Y_train_pred = GBDTree.predict(X_train)
	train_accu = np.sum(Y_train_pred == Y_train) / len(Y_train) * 100
	print('\nTraining accuracy: %.2f %%' %train_accu)

	Y_test_pred = GBDTree.predict(X_test)
	test_accu = np.sum(Y_test_pred == Y_test) / len(Y_test) * 100
	print('Testing accuracy: %.2f %%\n' %test_accu)


def main():
	run_model()


if __name__ == '__main__':
	main()