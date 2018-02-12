import numpy as np
import pandas as pd
import time


class KMeans:
    
    def __init__(self, k):
        self.k = k

    
    def train(self, data):
        self.data = data
        self.N = data.shape[0]
    
        # initialize mu
        self.mu = self.data.sample(self.k).values

        # alternating optimization
        converge = False
        while not converge:
            # cluters dictionary to store data in each cluster
            self.clusters = {i: [] for i in range(len(self.mu))}

            # partition
            for x in self.data.values:
                dist = [np.sum((x - center)**2) for center in self.mu]
                self.clusters[np.argmin(dist)].append(x)

            # update mu
            for j in self.clusters:
                new_mu = sum(self.clusters[j]) / len(self.clusters[j])
                if np.array_equal(new_mu, self.mu[j]):
                    self.mu[j] = new_mu
                else:
                    converge = True   # stop while converge

        return self.clusters, self.mu


    def clustering_error(self):
        error = 0
        for i in range(self.k):
            error += sum([np.sum((x - self.mu[i])**2) for x in self.clusters[i]])
        
        return error / self.N


def run_model():
	
	# load data
	data = pd.read_csv('data/hw8_nolabel_train.dat.txt', sep = ' ', header = None).dropna(axis = 1)

	# train model
	K = [2, 4, 6, 8, 10]
	experiments = 500

	start = time.clock()

	avg_err_in = []
	for k in K:
	    print('Training k-means with k = %d' %k)
	    
	    err = []
	    for i in range(experiments):
	        kmeans = KMeans(k)
	        kmeans.train(data)
	        err.append(kmeans.clustering_error())
	    
	    avg_err_in.append(np.mean(err))

	print('\nUsing %.2f seconds\n' % (time.clock() - start))


def main():
	run_model()


if __name__ == '__main__':
	main()