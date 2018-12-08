import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

'''
Ref:
    https://www.quora.com/How-does-a-SVM-choose-its-support-vectors
'''


def score(x,y,w):
  my_score = x @ w
  my_score = np.count_nonzero(np.sign(my_score) == np.sign(y))
  return my_score / y.shape[0]

def run(x,y):
    n,d = x.shape
    w = np.random.uniform(-1,1,(d,1))
    nb_epoch = 10 
    for epoch in range(nb_epoch):
        my_score = score(x,y,w)
        print("Score {} for epoch {}".format(my_score, epoch))

if __name__  == "__main__":
    X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    X1 = np.c_[np.ones((X.shape[0])), X] 
    y = np.vectorize(lambda x: 1 if x == 1 else -1)(y)
    y = y.reshape(-1,1)
    run(X1,y)
