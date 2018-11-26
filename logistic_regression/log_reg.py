import numpy as np
import pandas as pd
import pickle
import numpy
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score




## Logistic regression
def plot_points(xy, labels):
    for i, label in enumerate(set(labels)):
        points = np.array([xy[j,:] for j in range(len(xy)) if labels[j] == label])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        plt.scatter(points[:,0], points[:,1], marker=marker, color=color)
    plt.show()


def predict(w, x):
    z = x @ w
    sigmoid  = 1  / (1 + np.exp(- z))
    return sigmoid

# Cross entropy
def cost_function(w,x,y):
    cost = 0
    for sample in range(x.shape[0]):
        hx,  ys = predict(w,  x[sample,:]), y[sample]
        cost += ys * np.log(hx) + (1-ys) * np.log(1 - hx)
    return - 1/x.shape[0] *  cost


# Used for newton method
def hessian_matrix(x,y,w):
    cost = np.zeros((3,3))
    for sample in range(x.shape[0]):
        xx = x[sample,:].reshape(-1,1)
        yy = predict(w,xx.T)
        cost += xx @ yy * (1-yy) @ xx.T
    return cost

# Using Newton methods
def train_newton(x, y, lr=.015, eps=.01):
    ones = np.ones((x.shape[0], x.shape[1] -1))
    x = np.concatenate((x,ones),axis=-1)
    w = np.zeros((x.shape[1],1))
    y = y.reshape(-1,1)
    epoch = 100
    n = x.shape[0] 
    i = 1
    diff = eps + 1

    while i < epoch or diff > eps:
        i += 1  
        cost = cost_function(w,x,y)
        #print("Error function {}".format(cost))
        pred = predict(w,x)
        gradient = np.dot(x.T, pred - y)

        gradient /= n
        d2j = hessian_matrix(x,y,w)
        newton = np.linalg.inv(d2j) @ gradient
        w -= newton
        diff = np.linalg.norm(newton)

    # Prediction
    prediction = np.vectorize(lambda x: 0 if x < .5 else 1)(predict(w,x))
    succ  = 0
    for i in range(prediction.shape[0]):
        succ += prediction[i] == y[i][0]
    print("Acc {}/{}".format(succ, x.shape[0]))
    return w

def plot_descision_boundary(w,x,y):
    h = .01
    x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
    y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
    # xx, yy shape (144,221)
    # arange x => (221,)
    # arange y => (144,)
    xx, yy = np.meshgrid(np.arange(x_min, x_max,h), np.arange(y_min, y_max,h))
    
    xy_pairs = np.c_[xx.ravel(), yy.ravel()]
    ones = np.ones((xy_pairs.shape[0], 1))
    xy_pairs = np.concatenate([xy_pairs, ones], axis=-1)
    Z = predict(w,xy_pairs).reshape(xx.shape)
    Z = np.vectorize(lambda x: 1 if x > .5 else 0)(Z)

    # Plotting first point
    for i, label in enumerate(set(y)):
        points = np.array([x[j,:] for j in range(len(x)) if y[j] == label])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        plt.scatter(points[:,0], points[:,1], marker=marker, color=color)
    
    # Plotting the contour
    plt.contour(xx,yy,Z,colors='black')
    plt.show()




data = pd.read_csv('data.txt')

reduced = data[data['class'] <= 2]
X = reduced.as_matrix(columns=['alcohol', 'ash'])
y = label_binarize(reduced['class'].values, [1, 2])[:,0]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state = 44)
print("Xtrain shape {}".format(Xtrain.shape))
print("ytrain  shape {}".format(ytrain.shape))
print("Xtest shape {}".format(Xtest.shape))
print("ytest shape {}".format(ytest.shape))

# Plotting
MARKERS = ['+', 'x', '.']
COLORS = ['red', 'green', 'blue']


# ytrain = np.vectorize(lambda x: {0:-1,1:1}[x])(ytrain)
w = train_newton(Xtrain, ytrain)

# Test set
ones = np.ones((Xtest.shape[0], 1))
Xtest = np.concatenate((Xtest, ones),axis=-1)
predictions  = predict(w, Xtest)
predictions  = np.vectorize(lambda x: 1 if x > .5 else 0)(predictions)

print("Accurarcy: {}".format(accuracy_score(ytest, predictions)))
print("Precision: {}".format(precision_score(ytest, predictions, average='macro')))
print("Recall: {}".format(recall_score(ytest, predictions, average='macro')))
