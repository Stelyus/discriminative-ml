import numpy as np
import pandas as pd
import pickle
import numpy
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# Write code here
data = pd.read_csv('data.txt')

reduced = data[data['class'] <= 2]
X = reduced.as_matrix(columns=['alcohol', 'ash'])
y = label_binarize(reduced['class'].values, [1, 2])[:,0]

print("y shape {}".format(y.shape))
print("X shape{}".format(X.shape))


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state = 42)
Xtrain = Xtrain[:10,:]
ytrain = ytrain[:10]

#print('train:', len(Xtrain), 'test:', len(Xtest))



# Plotting
MARKERS = ['+', 'x', '.']
COLORS = ['red', 'green', 'blue']

def plot_points(xy, labels):
    for i, label in enumerate(set(labels)):
        points = np.array([xy[j,:] for j in range(len(xy)) if labels[j] == label])
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        plt.scatter(points[:,0], points[:,1], marker=marker, color=color)
    plt.show()

#plot_points(Xtrain[:10,:], ytrain[:10])

def gradient_descent(x, y, lr=.001):
    y = y.reshape(-1,1)
    w = np.zeros((x.shape[1]))
    alpha = np.zeros(y.shape)
    b = 0

    # Computing the derivatives
    for _ in range(100):
        dw = w - np.sum((alpha @ y.T) @ x)
        db = - np.sum(alpha @ y.T)
        dalpha = - np.sum(y.T @ ((x @ w) + b).reshape(-1,1) - 1)

        new_w = w - lr * dw
        new_b = b - lr * db
        new_alpha = alpha - lr * dalpha

        w, b, alpha = new_w, new_b, new_alpha

    count, success = 0, 0
    for i in range(x.shape[0]):
        success += int(np.sign(np.dot(x[i,:], w) + b)) == y[i][0]
        count += 1

    print("Succ: {}/{}".format(success, count))


ytrain = np.vectorize(lambda x: {0:-1,1:1}[x])(ytrain)

gradient_descent(Xtrain, ytrain)
