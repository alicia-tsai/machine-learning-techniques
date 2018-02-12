import numpy as np
import pandas as pd
import time
from DecisionTree import DecisionTree


class RandomForest:
    """Random forest using C&RT decision tree."""
    
    def __init__(self):
        self.trees_dict = {}     # internal tree dictionary 
    
    
    def construct_forest(self, data, col_y, size, max_height = 50):
        """Construct random forest by bagging from the original dataset."""
        
        self.N = data.shape[0]
        self.max_height = max_height
        
        print('Start training random forest')
        for i in range(size):
            resample_data = self.bagging(data, self.N)
            Tree = DecisionTree()
            Tree.construct_tree(resample_data, col_y, max_height = self.max_height)
            self.trees_dict[i] = Tree  
            
            if i % 1000 == 999:
                print('  ... training %d-th decision tree' %(i + 1))
                
        print('----------      END      -----------')
    
    
    def bagging(self, data, N):
        """Resample from the original dataset."""
        return data.sample(N, replace = True)
    
    
    def predict(self, x, trees_size=None):
        """ Predict the value of one data point x by uniform voting from all decision trees.
            Predict value using the first t trees can be done by setting trees size parameter.
        """

        if trees_size:
            vote = [self.trees_dict[i].predict(x) for i in range(trees_size)]
        else:
            vote = [Tree.predict(x) for Tree in self.trees_dict.values()]
        return 1 if np.sum(vote) >= 0 else -1


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
    T = 30000

    time_start = time.clock()
    RF = RandomForest()
    RF.construct_forest(data_train, col_y, size = T)

    print("\nUsing %.3f seconds" % (time.clock() - time_start))

    # model accuracy
    print('\n--- Random forest model accuarcy ---')
    print('Start predicting model...\n')
    time_start = time.clock()

    Y_train_pred = [RF.predict(x) for x in np.array(X_train)]
    train_acc = np.sum(Y_train_pred == Y_train) / len(Y_train) * 100
    print('Accuracy on the training set: %.2f %%' %train_acc)
    print("Using %.3f seconds to predict\n" % (time.clock() - time_start))

    time_start = time.clock()
    Y_test_pred = [RF.predict(x) for x in np.array(X_test)]
    test_acc = np.sum(Y_test_pred == Y_test) / len(Y_test) * 100
    print('Accuracy on the testing set: %.2f %%' %test_acc)
    print("Using %.3f seconds to predict\n" % (time.clock() - time_start))


def main():
    run_model()


if __name__ == '__main__':
    main()