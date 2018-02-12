import numpy as np
import pandas as pd
import time


class RBF_Network:
    """Regularized RBF Network using k-means."""
    
    def train(self, X, Y, k, gamma, lamb):
        
        N = X.shape[0]; self.gamma = gamma
        self.clusters, self.centers = self.run_kmeans(k, X)
        
        # feature transform from RBF
        Z = np.array([[self.RBF_kernel(x, mu, self.gamma) for mu in self.centers] for x in X.values])
        assert(Z.shape == (N, k))
        
        # compute beta of regularized full RBF Network
        inv = np.linalg.inv(np.dot(Z.T, Z) + np.identity(Z.shape[1]) * lamb)
        self.beta = inv.dot(Z.T).dot(Y)   
        
        return self.beta, self.centers
        
    
    def predict(self, X_test, binary_class = True):
        
        Z_test = np.array([[self.RBF_kernel(x, mu, self.gamma) for mu in self.centers] for x in X_test.values])
        return np.sign(Z_test.dot(self.beta)) if binary_class else Z_test.dot(self.beta)     
        
        
    def RBF_kernel(self, x, mu, gamma):
        return np.exp(-gamma * np.sum((x - mu)**2))
    
    
    def run_kmeans(self, k, data):
        data = data
        N = data.shape[0]
    
        # initialize mu
        mu = data.sample(k).values

        # alternating optimization
        converge = False
        while not converge:
            # cluters dictionary to store data in each cluster
            clusters = {i: [] for i in range(len(mu))}

            # partition
            for x in data.values:
                dist = [np.sum((x - center)**2) for center in mu]
                clusters[np.argmin(dist)].append(x)

            # update mu
            for j in clusters:
                new_mu = sum(clusters[j]) / len(clusters[j])
                if np.array_equal(new_mu, mu[j]):
                    mu[j] = new_mu
                else:
                    converge = True   # stop while converge

        return clusters, mu


def generate_data(file):
    
    data = pd.read_csv(file, sep = ' ', header = None)
    X = data.loc[:, :8]
    Y = np.array(data.loc[:, 9:])
    
    return X, Y


def run_model():

	# load data
	train_file = 'data/hw8_train.dat.txt'; test_file = 'data/hw8_test.dat.txt'
	X_train, Y_train = generate_data(train_file); X_test, Y_test = generate_data(test_file)

	# train model
	K = [2, 4, 6, 8, 10]               # for k-means clustering
	gamma = [0.001, 0.1, 1, 10, 100]   # for guassian kernel
	lamb = 0                           # for regularization

	start = time.clock()

	for k in K:
	    print('\nRBF Network Using k-Means with k = %d' %k)
	    RBFnet = RBF_Network()
	    
	    train_err = []; test_err = []
	    for r in gamma:
	        RBFnet.train(X_train, Y_train, k, r, lamb)
	        err_in = np.sum(Y_train == RBFnet.predict(X_train)) / len(Y_train)
	        err_out = np.sum(Y_test == RBFnet.predict(X_test)) / len(Y_test)
	        train_err.append(err_in); test_err.append(err_out)
	        
	        print('\t  Gamma = %.3f; Training error: %.2f%%; Testing error: %.2f%%' %(r, (err_in * 100),(err_out * 100)))

	print('Using %.4f seconds.\n' % (time.clock() - start))


def main():
	run_model()


if __name__ == '__main__':
	main()