import numpy as np
import pandas as pd
import pickle
import numpy
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


## HARD MARGIN

def plot_points(xy, labels):
    for i, label in enumerate(set(labels)):
        points = np.array([xy[j,:] for j in range(len(xy)) if labels[j] == label])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        plt.scatter(points[:,0], points[:,1], marker=marker, color=color)
    plt.show()

#plot_points(Xtrain[:10,:], ytrain[:10])

def acc(w, b, x, y):
    succ  = 0
    for sample in range(x.shape[0]):
        succ += y[sample][0] == np.sign(np.dot(w, x[sample,:]) - b)
    return succ / x.shape[0]
    
def gradient_descent(x, y, lr=.001):
    y = y.reshape(-1,1)
    w = np.random.uniform(0, 1, (x.shape[1]))
    alpha = np.zeros(y.shape)
    b = np.random.uniform(-1,1,1)
    EPOCH = 1_000

    # Computing the derivatives
    for _ in range(EPOCH):
        print("Accuracy {}".format(acc(w,b,x,y))) 
        
        '''
        for sample in range(x.shape[0]):
            val = y[sample] * (np.dot(x[sample,:], w) - b)
            if val < 1:
                w = w + lr * (y[sample] * x[sample,:] -2 * 1/EPOCH * w)
            else:
                w = w + lr * -2 * w * 1/EPOCH
        '''

    '''
    count, success = 0, 0
    for i in range(x.shape[0]):
        success += int(np.sign(np.dot(x[i,:], w) - b)) == y[i][0]
        count += 1

    print("Succ: {}/{}".format(success, count))
    '''


data = pd.read_csv('data.txt')

reduced = data[data['class'] <= 2]
X = reduced.as_matrix(columns=['alcohol', 'ash'])
y = label_binarize(reduced['class'].values, [1, 2])[:,0]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state = 42)
Xtrain = Xtrain[:10,:]
ytrain = ytrain[:10]

# Plotting
MARKERS = ['+', 'x', '.']
COLORS = ['red', 'green', 'blue']


ytrain = np.vectorize(lambda x: {0:-1,1:1}[x])(ytrain)
gradient_descent(Xtrain, ytrain)
