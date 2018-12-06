import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)


X1 = np.c_[np.ones((X.shape[0])), X] 

''' 
   X1  (50, 3)
   y   (50,)
'''
def perceptron_algorithm(x, y):
    y = np.vectorize(lambda x: 1 if x == 1 else -1)(y)
    y = y.reshape(-1,1) # 50,1
    # w = np.zeros((x.shape[1],1)) # 3,1
    w = np.random.uniform(-1,1,(x.shape[1],1))
    n = x.shape[0]
    epoch = 100
    for _ in range(epoch):
        for i in range(n):
            res = np.dot(w.reshape(3,), x[i])
            if np.sign(res) != np.sign(y[i]):
                w = w + .02 * np.sign(res) * res
        predict(x,y,w)
    return w

def predict(x,y,w):
    res = x @ w
    score = np.count_nonzero(np.sign(res) == np.sign(y))
    print("Score {}".format(score / res.shape[0]))


def plot(x,y,w):
    step = .01
    xmax, ymax = x[:,1].max() - 1, x[:,2].max() + 1
    xmin, ymin = x[:,1].min() - 1, x[:,2].min() - 1
    
    xx, yy = np.meshgrid(np.arange(xmin,xmax,step), np.arange(ymin,ymax,step))
    xy = np.c_[xx.ravel(), yy.ravel()]
    xy = np.c_[np.ones(xy.shape[0]), xy]
    # xy shape (1163052, 3)
    res = np.sign((xy @ w).reshape(xx.shape))
    plt.contour(xx,yy,res,colors='black')
    plt.scatter(X1[:,1], X1[:,2], marker='o', c=y)
    plt.axis([-5,10,-12,-1])
    plt.show()

w = perceptron_algorithm(X1, y)
plot(X1,y,w)
