#!/anaconda3/bin/python

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

'''
SMO algorithm

References:
    http://fourier.eng.hmc.edu/e176/lectures/ch9/node9.html
'''


class SVM():
    def __init__(self,x,y):
        n,d = x.shape
        
        self.x = x
        self.y = y
        
        # Init weight and bias
        self.intercept = 0
        self.w = np.random.uniform(-1,1,(d,1))
        
        # Positive lagrangian constraints
        self.alpha= np.zeros((n,1))
        return self

    def _alpha_optimization(self,a1,y1,a2,y2):
        if y1 == y2:
            U = max(0,a1 + a2)
            V = min(0, a1 + a2)
        else:
            U = max(0,a2 - a1)
            V = min(0,-a1 + a2)
            
        a2 = (U + V) / 2
        return a1, a2

    def run(self, nb_epoch=10):
        for epoch in range(nb_epoch):
            assert np.sum(self.alpha * self.y.T) == 0
            a1, a2 = self.alpha[:2].flatten()
            y1, y2 = self.y[:2].flatten()
            constant = y1 * a1 + y2 * a2
            a1, a2 = self._alpha_optimization(a1,y1,a2,y2)
            print("alpha {} {}".format(a1, a2))
            print("y {} {}".format(y1, y2))
            print("constant {}".format(constant))
            exit(0)


if __name__  == "__main__":
    X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    #X1 = np.c_[np.ones((X.shape[0])), X] 
    y = np.vectorize(lambda x: 1 if x == 1 else -1)(y)
    y = y.reshape(-1,1)
    print("X.shape {}".format(X.shape))
    print("y.shape {}".format(y.shape))
    print("-" * 15)
    svm = SVM(X,y)
    svm.run()
