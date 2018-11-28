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
    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
'''


class SoftmaxRegression():
    def __init__(self,x,y,K):
        global MARKERS
        global COLORS

        MARKERS = ['+', 'x', '.']
        COLORS = ['red', 'green', 'blue']
        
        N = x.shape[0]
        ones = np.ones((N, 1))
        self.x = np.concatenate([x, ones], axis=-1)
        self.y = y
        self.w = np.random.uniform(-1,1, (K, self.x.shape[1]))
        self.K = K


    def plot_boundaries(self,x,y):
        h = .01
        x_min, x_max  = np.min(x[:,0]) - 1, np.max(x[:,0]) + 1
        y_min, y_max = np.min(y[:,1]) - 1, np.max(x[:,1]) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        xy_pairs = np.c_[xx.ravel(),yy.ravel()]
        Z = self._predict(xy_pairs).reshape(xx.shape)
        
        y = y.argmax(axis=1)
        for i, label in enumerate(set(y)):
            points = np.array([x[j,:] for j in range(len(x)) if y[j] == label])
            marker = MARKERS[i % len(MARKERS)]
            color = COLORS[i % len(COLORS)]
            plt.scatter(points[:,0], points[:,1], marker=marker, color=color)
        
        plt.contour(xx, yy, Z, colors='black')
        plt.show()
            

    def _cost_function(self,x,y):
        n = x.shape[0] 
        
        u = np.exp(x @ self.w.T)
        v = np.sum(u, axis=-1).reshape(-1,1)
        
        softmax = u / v
        
        # To avoid -inf error
        softmax += 0.001
        return - 1/n * np.sum(np.diag(np.log(softmax) @ y.T))

    def _score(self,x,y):
        n = x.shape[0] 
        
        u = np.exp(x @ self.w.T)
        v = np.sum(u, axis=-1).reshape(-1,1)
        softmax = u / v
        true_label = y.argmax(axis=-1)
        predicted_label = softmax.argmax(axis=-1)
        prediction = np.count_nonzero(true_label == predicted_label)
        print("Prediction: {} / {}".format(prediction, y.shape[0]))
        

    def run(self):
        self._gradient()

    def _predict(self,x):
        ones = np.ones((x.shape[0], 1))
        x = np.concatenate((x,ones),axis=-1)
        
        u = np.exp(x @ self.w.T)
        v = np.sum(u, axis=-1).reshape(-1,1)
        
        softmax = u / v
        return softmax.argmax(axis=-1)

    # Gradient descent
    def _gradient(self, lr=.01):
        '''
                W structure
                n_class x n_weights
                w.shape (3,3)
        '''
        
        EPOCH = 110_000
        N = self.x.shape[0]
        self._score(self.x,self.y)
        
        for i in range(EPOCH):
            cost = self._cost_function(self.x,self.y)
            if i % 100 == 0:
                print("Iteration {0}, error cost: {1:.2f}".format(i, cost))
            
            u = np.exp(self.x @ self.w.T)
            v = np.sum(u, axis=-1).reshape(-1,1)
            softmax = u / v
            rhs = self.y - softmax
            gradient = np.zeros(self.w.shape)
            
            # Could use vstack but for genericity we use a for
            for j in range(0,self.K):
                gradient[j,:] = rhs[:,j] @ self.x
            
            gradient /= - N
            self.w -= lr * gradient

        self._score(self.x, self.y)
       

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

y = label_binarize(y, range(1,K+1))
sr = SoftmaxRegression(Xtrain, ytrain, K)
sr.run()
sr.plot_boundaries(X, y)
