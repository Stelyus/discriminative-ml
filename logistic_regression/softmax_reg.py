import numpy as np
import pandas as pd
import pickle
import numpy
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

#np.seterr(all='print')

'''
Softmax regression / Multinomial logistic regression

References:
    http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/
    http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
'''

class SoftmaxRegression():
    def __init__(self,x,y,K,opti='gradient',C=.00):
        '''
            x: data
            y: label
            K: number of class
            C: regularizer coefficients (L2)
            opti: Optimization choosen
            
            W structure
            n_class x n_weights
        '''
        
        global MARKERS
        global COLORS
        MARKERS = ['+', 'x', '.']
        COLORS = ['red', 'green', 'blue']
        
        N,d = x.shape
        ones = np.ones((N, 1))
        
        self.C = C
        self.y = y
        self.K = K
        self.opti = opti
        self.x = self._c_ones(x)
        self.w = np.random.uniform(-1,1, (K, self.x.shape[1]))
 
    def _c_ones(self,x):
        N = x.shape[0]
        ones = np.ones((N, 1))
        return np.concatenate([x, ones], axis=-1)
 
    def plot_boundaries(self,x,y):
        h = .01
        x_min, x_max  = np.min(x[:,0]) - 2, np.max(x[:,0]) + 2
        y_min, y_max = np.min(y[:,1]) - 2, np.max(x[:,1]) + 2
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
            
    
    def _softmax(self,x):
        '''
            Here we are substracting with max to prevent overflow
            But it doesnt prevent underflow
        ''' 
        exposant = x @ self.w.T
        exposant = exposant - np.max(exposant,axis=-1).reshape(-1,1) 
        
        u = np.exp(exposant)
        v = np.sum(u, axis=-1).reshape(-1,1)
        return u/v


    def _cost_function(self,x,y):
        n = x.shape[0] 
        softmax = self._softmax(x)
        
        # To avoid -inf error
        softmax += 0.001
        
        return - 1/n * np.sum(np.diag(np.log(softmax) @ y.T))

    def score(self,x,y):
        n = x.shape[0] 
        softmax = self._softmax(x)
        true_label = y.argmax(axis=-1)
        predicted_label = softmax.argmax(axis=-1)
        prediction = np.count_nonzero(true_label == predicted_label)
        return prediction / y.shape[0]
        

    def run(self, epoch=10_000):
        if self.opti == "gradient":
            self._gradient(epoch)
        else:
            raise ValueError("Unknown optimization")
    

    def _predict(self,x):
        x = self._c_ones(x)
        softmax = self._softmax(x)
        return softmax.argmax(axis=-1)

    # Gradient descent
    def _gradient(self, epoch, lr=.01):
        N = self.x.shape[0]
        acc = self.score(self.x,self.y)
        #print("Accurarcy train set: {0:.2f}".format(acc))
        
        for i in range(epoch):
            cost = self._cost_function(self.x,self.y)
            
            '''
            if i % 500 == 0:
                print("Iteration {0}, error cost: {1:.2f}".format(i, cost))
            '''
            
            softmax = self._softmax(self.x)
            # shape rhs 133,3
            rhs = self.y - softmax
            
            gradient = np.zeros(self.w.shape)
            
            # Same as np.vstack
            for j in range(0,self.K):
                gradient[j,:] =  rhs[:,j] @ self.x
                
            gradient /= - N
            # Add penalty
            gradient += self.C * self.w
            self.w -= lr * gradient

        acc = self.score(self.x, self.y)
        print("Accurarcy train set: {0:.2f}".format(acc))


def test_different_regularizer(C):
    sr = SoftmaxRegression(Xtrain, ytrain, K, C=0)
    sr.run()
    ones = np.ones((X.shape[0], 1))
    X_ones = np.concatenate([X,ones], axis=-1)

    ones_test= np.ones((Xtest.shape[0], 1))
    Xtest_ones = np.concatenate([Xtest,ones_test], axis=-1)
    #sr.plot_boundaries(Xtest, ytest)
    print("Accuracy test set: {0:.2f}".format(sr.score(Xtest_ones,ytest)))

# Number of class
K = 3

data = pd.read_csv('data.txt')
X = data.as_matrix(columns=['alcohol', 'flavanoids'])
#X = data.as_matrix()

#X = (X - np.mean(X,axis=0)) / np.std(X, axis=0)
y = data.as_matrix(columns=['class'])
y = label_binarize(y, range(1,K+1))
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25,
                                                random_state=42)
print("Xtrain shape {}".format(Xtrain.shape))
print("ytrain shape {}".format(ytrain.shape))

print("Xtest shape {}".format(Xtest.shape))
print("ytest shape {}".format(ytest.shape))

SLASH = 15
print("-" * SLASH)

#for C in [0,2,10,100]:
for C in [0,.01,1,10]:
    print("-" *SLASH)
    print("Testing for C={}".format(C))
    test_different_regularizer(C)
