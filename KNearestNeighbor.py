import numpy as np
import pandas as pd
import time


class KNN:
	"""K nearest neighbor model with two methods: 'knbor' with simple KNN and 'uniform' with RBF kernel."""
	
	def __init__(self, k):
		self.k = k
	
	
	def train(self, X, Y):
		"""Construct the training set for the KNN model."""
		
		self.X_train = X
		self.Y_train = Y
		self.N = self.X_train.shape[0]

		
	def knbor(self, x):
		"""Select the k nearest points and compute the score by uniform aggregation for one data point."""
		
		dist_index = [(np.sum((x - self.X_train[i])**2), i) for i in range(self.N)]
		dist_index.sort()
		k_neighbors = dist_index[:self.k]
		votes = []
		for neighbor in k_neighbors:
			index = neighbor[1]
			votes.append(self.Y_train[index])
		
		return 1 if np.sum(votes) >= 0 else -1
	
	
	def RBF_kernel(self, a, b, gamma):
		return np.exp(-gamma * np.sum((a - b)**2))
		
	
	def uniform(self, x, gamma):
		"""Select the k nearest points and compute the score by uniformly aggregating the RBF values for one data point."""

		dist_index = [(np.sum((x - self.X_train[i])**2), i) for i in range(self.N)]
		dist_index.sort()
		k_neighbors = dist_index[:self.k]
		votes = []
		for neighbor in k_neighbors:
			index = neighbor[1]
			score = self.RBF_kernel(x, self.X_train[index], gamma) * self.Y_train[index]
			votes.append(score)
			
		return 1 if np.sum(votes) >= 0 else -1
	
	
	def predict(self, X_test, method = 'knbor', gamma = None):
		"""Predict all data points in the data set with specifed method. Default method is 'knbor'."""
		
		Y_pred = []
		for x in X_test:
			pred = self.uniform(x, gamma) if method == 'uniform' else self.knbor(x)
			Y_pred.append(pred)
			
		return np.array(Y_pred)


def generate_data(file):
	
	data = pd.read_csv(file, sep = ' ', header = None)
	X = np.array(data.loc[:, :8])
	Y = np.array(data.loc[:, 9:])
	
	return X, Y


def run_model():
	
	# load data
	train_file = 'data/hw8_train.dat.txt'; test_file = 'data/hw8_test.dat.txt'
	X_train, Y_train = generate_data(train_file); X_test, Y_test = generate_data(test_file)

	# train model
	start = time.clock()
	K = [1,3, 5, 7, 9]
	gamma = [0.001, 0.1, 1, 10, 100]

	err_in_knbor = []
	err_out_knbor = []

	for k in K:
		print('\nKNN model with k = %d' %k)
		
		knn = KNN(k)
		knn.train(X_train, Y_train)
		
		print('\tSimple KNN')
		Y_train_pred = knn.predict(X_train)
		Y_test_pred = knn.predict(X_test)
		
		err_in = np.sum(Y_train_pred != Y_train.flatten()) / len(Y_train)
		err_out = np.sum(Y_test_pred != Y_test.flatten()) / len(Y_test)
		err_in_knbor.append(err_in); err_out_knbor.append(err_out)
		
		print('\t  Training error: %.2f%%; Testing error: %.2f%%' %((err_in * 100),(err_out * 100)))        
		
		
		print('\tRBF Kernel')
		err_in_uniform = []; err_out_uniform = []
		for r in gamma:
			Y_train_pred = knn.predict(X_train, method = 'uniform', gamma = r)
			Y_test_pred = knn.predict(X_test, method = 'uniform', gamma = r)
			
			err_in = np.sum(Y_train_pred != Y_train.flatten()) / len(Y_train)
			err_out = np.sum(Y_test_pred != Y_test.flatten()) / len(Y_test)
			err_in_uniform.append(err_in); err_out_uniform.append(err_out)
			
			print('\t  Gamma = %.3f; Training error: %.2f%%; Testing error: %.2f%%' %(r, (err_in * 100),(err_out * 100)))    
		
		
	print('\nUsing %.2f seconds\n' % (time.clock() - start))


def main():
	run_model()


if __name__ == '__main__':
	main()