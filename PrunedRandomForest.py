import numpy as np
import pandas as pd
import time
from RandomForest import RandomForest, generate_data


def run_model():
	
	# load data
	train_file = 'data/hw7_train.dat.txt'; test_file = 'data/hw7_test.dat.txt'
	data_train = pd.read_csv(train_file, sep = ' ', header = None, names=[0, 1, 'y'])
	data_test = pd.read_csv(test_file, sep = ' ', header = None, names=[0, 1, 'y'])
	X_train, Y_train = generate_data(train_file); X_test, Y_test = generate_data(test_file)
	
	# train model
	col_y = 'y'
	T = 30000; max_height = 1

	time_start = time.clock()
	RF_Prune = RandomForest()
	RF_Prune.construct_forest(data_train, col_y, size = T, max_height = max_height)

	print("Using %.3f seconds" % (time.clock() - time_start))

	# model accuracy
	print('\n--- Pruned Random forest model accuarcy ---')

	Y_train_pred = [RF_Prune.predict(x) for x in np.array(X_train)]
	train_acc = np.sum(Y_train_pred == Y_train) / len(Y_train) * 100
	print('Model accuracy on the training set: %.2f %%' %train_acc)

	Y_test_pred = [RF_Prune.predict(x) for x in np.array(X_test)]
	test_acc = np.sum(Y_test_pred == Y_test) / len(Y_test) * 100
	print('Accuracy on the testing set: %.2f %%\n' %test_acc)


def main():
	run_model()


if __name__ == '__main__':
	main()