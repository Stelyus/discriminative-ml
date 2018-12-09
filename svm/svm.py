#!/anaconda3/bin/python

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

'''
Hard margin only

Ref:
    https://www.quora.com/How-does-a-SVM-choose-its-support-vectors
    https://people.cs.pitt.edu/~milos/courses/cs2750-Spring03/lectures/class11.pdf
    http://www.csc.kth.se/utbildning/kth/kurser/DD2427/bik12/DownloadMaterial/Lectures/Lecture9.pdf
'''

def score(x,y,w):
  my_score = x @ w
  my_score = np.count_nonzero(np.sign(my_score) == np.sign(y))
  return my_score / y.shape[0]


'''
    alphas n,1
    x      n,d
    y      n,1
    w      d,1
    
'''
def run(x,y,nb_epoch=10):
    n,d = x.shape
    # Init weight and bias
    w, b = np.random.uniform(-1,1,(d,1)), 0
    # Lagrangian constraints
    alphas  = np.random.uniform(0,1,(n,1))
    for epoch in range(nb_epoch):
        my_score = score(x,y,w)
        res = alphas * y * x
        w = np.sum(res, axis=0).reshape(-1,1)
        print("Score {} for epoch {}".format(my_score, epoch))
    return w,b 

if __name__  == "__main__":
    X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    #X1 = np.c_[np.ones((X.shape[0])), X] 
    y = np.vectorize(lambda x: 1 if x == 1 else -1)(y)
    y = y.reshape(-1,1)
    print("X.shape {}".format(X.shape))
    print("y.shape {}".format(y.shape))
    print("-" * 10)
    run(X,y)
