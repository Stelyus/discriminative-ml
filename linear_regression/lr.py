import numpy as np
import pandas as pd
import pickle
import numpy
import matplotlib.pyplot as plt


# Loading data
fin = open('data.pkl', 'rb')
x = pickle.load(fin)
y = pickle.load(fin)
fin.close()

def show(x,y):
    plt.plot(x, y, 'x')
    plt.plot(x, x * 0.06388117 + 0.75016254)
    plt.xlabel('Age')
    plt.ylabel('Height')
    plt.show()

# show(x,y)

def mse(x,y,w,b):
    n = x.shape[0]
    return 1/(2*n) * np.sum(((x@w+b) - y) ** 2)

x = x.reshape(-1,1)
y = y.reshape(-1,1)

def gradient_descent(x, y, lr=.01):
    N, nb_var = x.shape
    w, b = np.zeros((nb_var, 1)) , 0

    for i in range(10000):
        error = mse(x,y,w, b)
        print("Error %s" % error)
        # Using the chain rule derivative
        db = 1/N * np.sum((x@w+b)-y)
        dw = 1/N * np.sum(x*((x@w+b)-y))
        
        w = w - lr * dw
        b = b - lr * db

    print("My solution x * {} + {}".format(w[0][0], b))
    print("Solution x * 0.06388117 + 0.75016254")



gradient_descent(x,y)
