#!/anaconda3/bin/python

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

'''
SMO algorithm

References:
    http://fourier.eng.hmc.edu/e176/lectures/ch9/node6.html
    http://fourier.eng.hmc.edu/e176/lectures/ch9/node9.html
    https://pdfs.semanticscholar.org/59ee/e096b49d66f39891eb88a6c84cc89acba12d.pdf?_ga=2.115831941.287013835.1544789908-1848930882.1544789908
    '''

class SVM(object):
    def __init__(self,x,y,C=1):
        n,d = x.shape

        self.x = x
        self.y = y
        self.C = C
        
        # Init weight and bias
        self.intercept = 0
        self.w = np.random.uniform(-1,1,(d,1))
    
        # Positive lagrangian constraints
        self.neg_arg, self.pos_arg = y == -1, y == 1
        neg_count = np.count_nonzero(self.neg_arg)
        pos_count = np.count_nonzero(self.pos_arg)
        
        self.alpha = np.zeros((n,1))
        self.alpha[self.neg_arg] = 1/neg_count
        self.alpha[self.pos_arg] = 1/pos_count
        
        # Generating gram matrix
        self.gram_matrix = self.x @ self.x.T


    def _assert_linear_contraint(self):
        ret = np.diag(self.alpha @ self.y.T).reshape(-1,1)
        assert np.sum(ret[self.pos_arg]) == -np.sum(ret[self.neg_arg])


    def _prediction_optimization(self,xi):
        xi = xi.reshape(-1,1)
        gram_matrix = self.x @ xi
        lhs = np.diag(self.y @ self.alpha.T)
        
        ret = lhs.T @ gram_matrix + self.intercept
        return ret

    def _alpha_optimization(self,i,j):
        # 50,
        tmp = self.alpha.reshape(-1) * self.y.reshape(-1) 
        ret = tmp * self.x.T
       
        alpha_flatten = self.alpha.reshape(-1)
        y_flatten = self.y.reshape(-1)
        a1, a2 = alpha_flatten[i], alpha_flatten[j]
        x1, x2 = self.x[i], self.x[j]
        y1, y2 = y_flatten[i], y_flatten[j]
        
        # Computing the bounds
        if y1 != y2:
            L = max(0, a2 - a1)
            H = min(self.C, self.C + a2 - a1)
        else:
            L = max(0, a2 + a1 - self.C)
            H = min(self.C, a2 + a1)

        # The second derivative of the objective function along the diagonal
        # line can be expressed as
        n = np.dot(x1,x1) + np.dot(x2,x2) - 2 * np.dot(x1, x2)
        assert n >= 0

        #TODO: May change to "- self.intercept"
        e1 = self._prediction_optimization(x1)
        e2 = self._prediction_optimization(x2)
        e1 = y1 * a1 * np.dot(x1,x1) + self.intercept
        a2_new = a2 + (y2 * (e1 - e2)) / n
        
        if a2_new >= H:
           a2_new = H
        elif a2_new <= L:
           a2_new = L

        a1_new = a1 + y1 * y2 * (a2 - a2_new)

        print("Old value {} {}".format(a1,a2))
        print("New value {} {}".format(a1_new, a2_new))

        self.alpha[i] = a1_new
        self.alpha[j] = a2_new

        return a1_new, a2_new
    
    def run(self, nb_epoch=10):
        for epoch in range(nb_epoch):
            self._assert_linear_contraint()
            self._alpha_optimization(0,1)
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
