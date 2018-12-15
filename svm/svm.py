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
    # Soft margin
    def __init__(self,x,y,C=1):
        n,d = x.shape

        self.x = x
        self.y = y
        self.C = C

        # Init weight and bias
        #TODO: Update intercept
        self.intercept = 0
        self.w = np.random.uniform(-1,1,(d,1))
    
        # Positive lagrangian constraints
        self.neg_arg, self.pos_arg = y == -1, y == 1
        neg_count = np.count_nonzero(self.neg_arg)
        pos_count = np.count_nonzero(self.pos_arg)
        
        self.alpha = np.zeros((n,1))
        # Initilizing them with non zero value while respecting the linear
        # constraint
        self.alpha[self.neg_arg] = 1/neg_count
        self.alpha[self.pos_arg] = 1/pos_count

        self.max_digit = max(len(str(neg_count)), len(str(pos_count)))

    
    # Using decimal library to avoid floating point approximation
    def _sum(self, arr):
        n = arr.shape[0]
        ret = 0
        power = np.power(10, self.max_digit)
        for i in range(n):
            ret +=  arr[i] * power
        return ret

    def _assert_linear_constraint(self):
        ret = np.diag(self.alpha @ self.y.T).reshape(-1,1)

        pos_sum = self._sum(ret[self.pos_arg])
        neg_sum = - self._sum(ret[self.neg_arg])
    
        assert pos_sum == neg_sum
    
    def _prediction_optimization(self,xi):
        xi = xi.reshape(-1,1)
        lhs = np.diag(self.y @ self.alpha.T)
        print(lhs)
        w  = np.sum((lhs * self.x.T).T, axis=0)
        ret = np.dot(w, xi) + self.intercept
        return ret

    # Heuristics for choosing which multipliers to optimize
    def _heuristics(self):
        n, d = self.x.shape 
        x1, x2 = None, None
        
        for i in range(n):
            ret = self._prediction_optimization(self.x[i]) - y[i]
            ai = self.alpha[i] 

            print("-- Heuristics --")
            print(ret)
            print(ai)
            
            # Find two alphas which violates the KKT conditions
            if (ret >= 1 and ai != 0) \
                or (ret == 1 and (ai <= 0 or ai >= self.C)) \
                or (ret <= 1 and ai != self.C):
                if x1 is not None:
                    x2 = i
                    return x1, x2
                else:
                    x1 = i
        
        # Not found
        return -1,-1
    
    def _alpha_optimization(self,i,j):
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
        
        # n should be positive for mathematical convenience
        assert n >= 0

        e1 = self._prediction_optimization(x1) - y1
        e2 = self._prediction_optimization(x2) - y2
        a2_new = a2 + (y2 * (e1 - e2)) / n
        
        if a2_new >= H:
           a2_new = H
        elif a2_new <= L:
           a2_new = L

        a1_new = a1 + y1 * y2 * (a2 - a2_new)
        
        b1 = self.intercept - e1 + y1 * (a1_new - a1) * np.dot(x1,x1) \
            - y2 * (a2_new -a2) * np.dot(x1,x2)

        b2 = self.intercept - e2 + y1 * (a1_new - a1) * np.dot(x1,x2) \
            - y2 * (a2_new -a2) * np.dot(x2,x2)

        if a1_new > 0 and a1_new < self.C:
            self.intercept = b1
        elif a2_new > 0 and a2_new < self.C:
            self.intercept = b2
        else:
            self.intercept = (b1 + b2) / 2
            
        print("Optimizing {} {}".format(i,j))
        print("y  value {} {}".format(y1,y2))
        print("Old alpha value {} {}".format(a1,a2))
        print("New alpha value {} {}\n".format(a1_new, a2_new))
        print("Prediction value {} {}".format(e1,e2))
        print("Intercept value {}".format(self.intercept))

        self.alpha[i] = a1_new
        self.alpha[j] = a2_new
    
    # Plotting the support vectors
    def _plot_sv(self): 
         arg = np.where(self.alpha.reshape(-1) != 0.)
         plt.scatter(self.x[:,0],self.x[:,1],c=self.y.reshape(-1))
         plt.scatter(self.x[arg,0],self.x[arg,1],c='red')
         plt.show()
        
    def run(self):
        # TODO: Add tolerance
        for _ in range(1000):
            # Checking the linear constraint sum alpha yi = 0
            self._assert_linear_constraint()
            # Getting the heuristics alphas
            i,j = self._heuristics()
            if i == j and i == -1:
                break
            # Optimize it
            self._alpha_optimization(i,j)
        self._plot_sv()

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
