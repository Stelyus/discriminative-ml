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
Logistic regression
References:
    https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
    https://stats.stackexchange.com/questions/278771/how-is-the-cost-function-from-logistic-regression-derivated
'''

class LogisticRegression():
    def __init__(self,x,y,opti='gradient'):
        ones = np.ones((x.shape[0], x.shape[1] -1))
        x = np.concatenate((x,ones),axis=-1)
        
        # Plotting
        global MARKERS, COLORS
        
        MARKERS = ['+', 'x', '.']
        COLORS = ['red', 'green', 'blue']
        
        self.y = y.reshape(-1,1)
        self.x = x
        self.w = np.zeros((x.shape[1],1))
        self.opti = opti

    def _predict(self, x):
        z = x @ self.w
        sigmoid  = 1  / (1 + np.exp(- z))
        return sigmoid

    # Cross entropy
    def _cost_function(self):
        N, cost= self.x.shape[0], 0
        for sample in range(N):
            hx,  ys = self._predict(self.x[sample,:]), self.y[sample]
            cost += ys * np.log(hx) + (1-ys) * np.log(1 - hx)
        return - 1/N *  cost

    # Used for newton method
    def _hessian_matrix(self):
        N, cost = self.x.shape[0], np.zeros((3,3))
        for sample in range(N):
            xx = self.x[sample,:].reshape(-1,1)
            yy = self._predict(xx.T)
            cost += xx @ yy * (1-yy) @ xx.T
        return cost


    def plot_descision_boundary(self,x,y):
        h = .01
        x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
        y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max,h), np.arange(y_min, y_max,h))
        
        xy_pairs = np.c_[xx.ravel(), yy.ravel()]
        ones = np.ones((xy_pairs.shape[0], 1))
        xy_pairs = np.concatenate([xy_pairs, ones], axis=-1)
        Z = self._predict(xy_pairs).reshape(xx.shape)
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

        
    # Using Newton methods
    def run(self,lr=.015, eps=.01, epoch=1_000):
        if self.opti == 'newton':
            self._newton(lr, eps, epoch)
        elif self.opti == 'gradient':
            self._gradient(lr, eps, epoch)
        else:
            raise ValueError("Wrong optimization")
    

    def _gradient(self,lr,eps,epoch):
        n = self.x.shape[0]
        diff = eps + 1
        mapping = np.vectorize(lambda x: 0 if x < .5 else 1)

        prediction = mapping(self._predict(self.x))
        succ = np.count_nonzero(prediction.reshape(-1) == self.y.reshape(-1))
        print("Acc {}/{}".format(succ, n))
        
        for _ in range(epoch):
            cost = self._cost_function()
            print("Error function {}".format(cost))
            pred = self._predict(self.x)
            gradient = np.dot(self.x.T, pred - self.y)
            
            gradient /= n
            gradient *= lr
            self.w -= gradient
        
        # Prediction
        prediction = mapping(self._predict(self.x))
        succ = np.count_nonzero(prediction.reshape(-1) == self.y.reshape(-1))
        print("Acc {}/{}".format(succ, n))


    def _newton(self, lr, eps, epoch):
        n, i = self.x.shape[0], 1
        diff = eps + 1

        mapping = np.vectorize(lambda x: 0 if x < .5 else 1)

        prediction = mapping(self._predict(self.x))
        succ = np.count_nonzero(prediction.reshape(-1) == self.y.reshape(-1))
        print("Acc {}/{}".format(succ, n))
        
        while i < epoch or diff > eps:
            i += 1
            cost = self._cost_function()
            #print("Error function {}".format(cost))
            pred = self._predict(self.x)
            gradient = np.dot(self.x.T, pred - self.y)

            gradient /= n
            d2j = self._hessian_matrix()
            newton = np.linalg.inv(d2j) @ gradient
            self.w -= newton
            diff = np.linalg.norm(newton)

        # Prediction
        prediction = mapping(self._predict(self.x))
        succ = np.count_nonzero(prediction.reshape(-1) == self.y.reshape(-1))
        print("Acc {}/{}".format(succ, n))

data = pd.read_csv('data.txt')

reduced = data[data['class'] <= 2]
X = reduced.as_matrix(columns=['alcohol', 'ash'])
y = label_binarize(reduced['class'].values, [1, 2])[:,0]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state = 44)
print("Xtrain shape {}".format(Xtrain.shape))
print("ytrain  shape {}".format(ytrain.shape))
print("Xtest shape {}".format(Xtest.shape))
print("ytest shape {}".format(ytest.shape))


lg = LogisticRegression(Xtrain, ytrain, opti='gradient')
lg.run()
lg.plot_descision_boundary(X, y)

'''

print("Accurarcy: {}".format(accuracy_score(ytest, predictions)))
print("Precision: {}".format(precision_score(ytest, predictions, average='macro')))
print("Recall: {}".format(recall_score(ytest, predictions, average='macro')))
'''
