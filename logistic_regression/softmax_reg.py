import numpy as np
import pandas as pd
import pickle
import numpy
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

'''
Softmax regression / Multinomial logistic regression

References:
    http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
'''

MARKERS = ['+', 'x', '.']
COLORS = ['red', 'green', 'blue']


'''
    xi.shape (3,)
'''  
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



def cost_refractor(x,y,w,K):
    n = x.shape[0] 
    # value = softmax  shape (133,3)
    u = np.exp(x @ w.T)
    v = np.sum(u, axis=-1).reshape(-1,1)
    softmax = u / v
    return - 1/n * np.sum(np.diag(np.log(softmax) @ y.T))

# Gradient descent
def softmax_reg(x,y,K,lr=.01):
    EPOCH = 100
    N = x.shape[0]
    ones = np.ones((N, 1))
    x = np.concatenate([x, ones], axis=-1)
    

    '''
    n_class x n_weights
    w.shape (3,3)
    '''
    
    w = np.random.uniform(-1,1, (K, x.shape[1]))

    for _ in range(EPOCH):
        cost = cost_function(x,y,w,K)
        cost1 = cost_function_refractor(x,y,w,K)

        print("Error cost: {0:.2f}".format(cost_refractor))
        gradient_dj = 0
        
        '''
            yi, xi = y[sample,:], x[sample,:]
            yi = yi.reshape(-1,1) 
            theta_t = w.T @ yi
            u = np.exp(theta_t.T @ xi)
            v = np.sum(np.exp(w @ xi))
            np.log(u / v)
        '''
        
        w = w - lr * gradient_dj
        sys.exit()

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

#plot_points(Xtrain, ytrain.argmax(axis=1))
softmax_reg(Xtrain, ytrain, K)
