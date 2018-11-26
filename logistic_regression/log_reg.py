import numpy as np
import pandas as pd
import pickle
import numpy
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


## Logistic regression

def plot_points(xy, labels):
    for i, label in enumerate(set(labels)):
        points = np.array([xy[j,:] for j in range(len(xy)) if labels[j] == label])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        plt.scatter(points[:,0], points[:,1], marker=marker, color=color)
    plt.show()


# x.shape 10,2
# w.shape 2 1
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


def hessian_matrix(x,y,w):
    yhat = predict(w,x)
    ret = x.T.dot(np.diag(np.multiply(yhat, (1 - yhat)).reshape(x.shape[0]))).dot(x)
    return ret

def gradient_descent(x, y, lr=.015):
    ones = np.ones((x.shape[0], x.shape[1] -1))
    x = np.concatenate((x,ones),axis=-1)
    w = np.zeros((x.shape[1],1))
    y = y.reshape(-1,1)
    epoch = 10_000
    n = x.shape[0] 


    for i in range(epoch):
        cost = cost_function(w,x,y)
        print("Error function {}".format(cost))
        pred = predict(w,x)
        gradient = np.dot(x.T, pred - y)

        gradient /= n
        d2j = hessian_matrix(x,y,w)
        newton = np.linalg.inv(d2j) @ gradient
        w -= newton


    # Prediction
    prediction = np.vectorize(lambda x: 0 if x < .5 else 1)(predict(w,x))
    succ  = 0
    for i in range(prediction.shape[0]):
        succ += prediction[i] == y[i][0]
    print("Acc {}/{}".format(succ, x.shape[0]))




data = pd.read_csv('data.txt')

reduced = data[data['class'] <= 2]
X = reduced.as_matrix(columns=['alcohol', 'ash'])
y = label_binarize(reduced['class'].values, [1, 2])[:,0]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state = 42)


# Plotting
MARKERS = ['+', 'x', '.']
COLORS = ['red', 'green', 'blue']


# ytrain = np.vectorize(lambda x: {0:-1,1:1}[x])(ytrain)
gradient_descent(Xtrain, ytrain)
