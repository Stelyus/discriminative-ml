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
    n_class x n_sample
    w.shape (3,3)
    xi.shape (3,)
'''  

def cost_function(x,y,w,K):
    cost = 0
    for sample in range(x.shape[0]):
        yi, xi = y[sample,:], x[sample,:]
        yi = yi.reshape(-1,1) 
        theta_t = w.T @ yi
        u = np.exp(theta_t.T @ xi)
        v = np.sum(np.exp(w @ xi))
        cost += np.log(u / v)
    return - cost
            



def softmax_reg(x,y,K):
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate([x, ones], axis=-1)
    w = np.random.uniform(-1,1, (K, x.shape[1]))
    
    cost = cost_function(x,y,w,K)
    print("Error: {}".format(cost))



def plot_points(xy, labels):
    for i, label in enumerate(set(labels)):
        points = np.array([xy[j,:] for j in range(len(xy)) if labels[j] == label])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        plt.scatter(points[:,0], points[:,1], marker=marker, color=color)
    plt.show()

K = 3
data = pd.read_csv('data.txt')
X = data.as_matrix(columns=['alcohol', 'flavanoids'])
y = data.as_matrix(columns=['class'])
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)
ytrain = label_binarize(ytrain, range(1,K+1))

print("Xtrain shape {}".format(Xtrain.shape))
print("ytrain shape {}".format(ytrain.shape))

print("Xtest shape {}".format(Xtest.shape))
print("ytest shape {}".format(ytest.shape))

# Number of class
#plot_points(Xtrain, ytrain.argmax(axis=1))
softmax_reg(Xtrain, ytrain, K)
