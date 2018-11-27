import numpy as np
import pandas as pd
import pickle
import numpy
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

np.seterr(all='print')

'''
Softmax regression / Multinomial logistic regression

References:
    http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
'''

MARKERS = ['+', 'x', '.']
COLORS = ['red', 'green', 'blue']


'''
Not used anymore
def cost_function(x,y,w,K):
    cost, N = 0, x.shape[0]
    for sample in range(N):
        yi, xi = y[sample,:], x[sample,:]
        yi = yi.reshape(-1,1) 
        theta_t = w.T @ yi
        u = np.exp(theta_t.T @ xi)
        v = np.sum(np.exp(w @ xi))
        cost += np.log(u / v)
    return - 1/N * cost
'''



def cost_refractor(x,y,w,K):
    n = x.shape[0] 
    
    print(np.max(x))
    print(np.min(x))
    print(w)
    u = np.exp(x @ w.T)
    v = np.sum(u, axis=-1).reshape(-1,1)
    
    softmax = u / v
    softmax += 0.001
    if np.isnan(np.log(softmax)).any():
        print("Found it")
        sys.exit()
    return - 1/n * np.sum(np.diag(np.log(softmax) @ y.T))

# Gradient descent
def softmax_reg(x,y,K,lr=.01,precision=.01):
    EPOCH = 1000
    N = x.shape[0]
    ones = np.ones((N, 1))
    x = np.concatenate([x, ones], axis=-1)
    i = 1
    # delta = precision + 1
    
    '''
            n_class x n_weights
            w.shape (3,3)
    '''
    w = np.random.uniform(-1,1, (K, x.shape[1]))

    while i < EPOCH:
        cost = cost_refractor(x,y,w,K)
        print("Iteration {0}, error cost: {1:.2f}".format(i, cost))
        u = np.exp(x @ w.T)
        v = np.sum(u, axis=-1).reshape(-1,1)
        # softmax (133,3)
        softmax = u / v
        
        # p (133,3)
        p = y - softmax
        
        #res (133,133)
        res = x @ p.T
        
        gradient = -1/N * np.sum(np.diag(res))
        w = w - lr * gradient
        i += 1 
    return w
   



def plot_points(xy, labels):
    for i, label in enumerate(set(labels)):
        points = np.array([xy[j,:] for j in range(len(xy)) if labels[j] == label])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        plt.scatter(points[:,0], points[:,1], marker=marker, color=color)
    plt.show()


# Number of class
K = 3

data = pd.read_csv('data.txt')
X = data.as_matrix(columns=['alcohol', 'flavanoids'])
y = data.as_matrix(columns=['class'])
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)
ytrain = label_binarize(ytrain, range(1, K+1))

print("Xtrain shape {}".format(Xtrain.shape))
print("ytrain shape {}".format(ytrain.shape))

print("Xtest shape {}".format(Xtest.shape))
print("ytest shape {}".format(ytest.shape))
print("-" * 15)

#plot_points(Xtrain, ytrain.argmax(axis=1))
softmax_reg(Xtrain, ytrain, K)
