import numpy as np
import pandas as pd
import pickle
import numpy
import sys
import matplotlib.pyplot as plt
'''
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
'''



np.seterr(all='print')

'''
Softmax regression / Multinomial logistic regression

References:
    http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
'''

MARKERS = ['+', 'x', '.']
COLORS = ['red', 'green', 'blue']

def score(x,y,w, K):
    n = x.shape[0] 
    
    u = np.exp(x @ w.T)
    v = np.sum(u, axis=-1).reshape(-1,1)
    softmax = u / v
    true_label = y.argmax(axis=-1)
    predicted_label = softmax.argmax(axis=-1)
    prediction = np.count_nonzero(true_label == predicted_label)
    print("Prediction: {} / {}".format(prediction, y.shape[0]))
    


def predict(x,w):
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((x,ones),axis=-1)
    
    u = np.exp(x @ w.T)
    v = np.sum(u, axis=-1).reshape(-1,1)
    
    softmax = u / v
    return softmax.argmax(axis=-1)


def cost_function(x,y,w,K):
    n = x.shape[0] 
    
    u = np.exp(x @ w.T)
    v = np.sum(u, axis=-1).reshape(-1,1)
    
    softmax = u / v
    softmax += 0.001
    return - 1/n * np.sum(np.diag(np.log(softmax) @ y.T))

# Gradient descent
def softmax_reg(x,y,K,lr=.01):
    '''
            W structure
            n_class x n_weights
            w.shape (3,3)
    '''
    
    EPOCH = 100_000
    N = x.shape[0]
    ones = np.ones((N, 1))
    x = np.concatenate([x, ones], axis=-1)
    w = np.random.uniform(-1,1, (K, x.shape[1]))
    score(x,y,w,K)
    
    for i in range(EPOCH):
        cost = cost_function(x,y,w,K)
        if i % 100 == 0:
            print("Iteration {0}, error cost: {1:.2f}".format(i, cost))
        
        u = np.exp(x @ w.T)
        v = np.sum(u, axis=-1).reshape(-1,1)
        softmax = u / v
        rhs = y - softmax
        gradient = np.zeros(w.shape)
        
        # Could use vstack but for genericity we use a for
        for j in range(0,K):
            gradient[j,:] = rhs[:,j] @ x
        
        gradient /= - N
        w -= lr * gradient

    score(x,y,w,K)
    return w
   
def plot_boundaries(x,y,w, h=.1):
    x_min, x_max  = np.min(x[:,0]) - 1, np.max(x[:,0]) + 1
    y_min, y_max = np.min(y[:,1]) - 1, np.max(x[:,1]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xy_pairs = np.c_[xx.ravel(),yy.ravel()]
    Z = predict(xy_pairs, w).reshape(xx.shape)
    
    y = y.argmax(axis=1)
    for i, label in enumerate(set(y)):
        points = np.array([x[j,:] for j in range(len(x)) if y[j] == label])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        plt.scatter(points[:,0], points[:,1], marker=marker, color=color)

    plt.contour(xx, yy, Z, colors='black')
    plt.show()
        


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

w = softmax_reg(Xtrain, ytrain, K)
plot_boundaries(Xtrain, ytrain, w)
