import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets.samples_generator import make_blobs


'''
References:
    http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf
'''

X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
X1 = np.c_[np.ones((X.shape[0])), X] 

def perceptron_algorithm(x, y):
    y = y.reshape(-1,1) # 50,1
    points, ret = [], []
    w = np.random.uniform(-1,1,(x.shape[1],1))
    n, lr, nb_epoch = x.shape[0], .01, 100
    for epoch in range(nb_epoch):
        for i in range(n):
            res = np.dot(w.reshape(3,),x[i])
            if np.sign(res) != np.sign(y[i]):
                ret.append(w)
                points.append(i)
                
                #print("w {}".format(w))
                #print("y {}, res {}".format(y[i], res))
                #print("input {}".format(x[i]))
                
                #plot(x,y.reshape(-1),w,i)
                w = update_weight(w,res,x[i],y[i],lr=lr)
                #print("new weights: {}".format(w))
                #plot(x,y.reshape(-1),w,i)
    
    # ret.append(w)
    return ret, points

def update_weight(weight, res, inp, y, lr=.01):
    error = y - res
    update  = (lr * error * inp).reshape(-1,1)
    print("update value {}".format(update))
    return weight + update

def animation_perceptron(x,y):
    def animate(iteration):
        plt.clf()
        step = .01
        w = ret[iteration].reshape(-1,1)
        i = points[iteration]
        pt = x[i]
        res = np.dot(w.reshape(3,),x[i])
        
        print("w {}".format(w))
        print("y {}, res {}".format(y[i], res))
        w = update_weight(w,res,pt,y[i])
        print("new weights: {}".format(w))
        xmax, xmin = 11, -5
        ymax, ymin = 5, -13
        
        xx, yy = np.meshgrid(np.arange(xmin,xmax,step), np.arange(ymin,ymax,step))
        xy = np.c_[xx.ravel(), yy.ravel()]
        xy = np.c_[np.ones(xy.shape[0]), xy]
        res = np.sign((xy @ w).reshape(xx.shape))
        
        cont = plt.contour(xx,yy,res,colors='black')
        plt.quiver(0,0,w[1],w[2])
        plt.scatter(x[:,1], x[:,2], marker='o', c=y)
        plt.scatter(pt[1], pt[2],c='red')
        
        plt.axis([-5,10,-12,5])
        return cont
    
    fig = plt.figure()
    ret, points = perceptron_algorithm(x,y)
    ani = animation.FuncAnimation(fig, animate, frames=len(ret),
                                  blit=False, interval=1000, repeat=False)
    plt.show()

def predict(x,y,w):
    correct_pred = np.count_nonzero(np.sign(x @ w) == np.sign(y))
    accuracy  = correct_pred / x.shape[0]
    print("Accuracy {}%".format(accuracy))

def plot(x,y,w,i):
    step = .01
    res = np.dot(w.reshape(3,),x[i])
    xmax, xmin = 11, -5
    ymax, ymin = 5, -13
    
    xx, yy = np.meshgrid(np.arange(xmin,xmax,step), np.arange(ymin,ymax,step))
    xy = np.c_[xx.ravel(), yy.ravel()]
    xy = np.c_[np.ones(xy.shape[0]), xy]
    res = np.sign((xy @ w).reshape(xx.shape))
    
    cont = plt.contour(xx,yy,res,colors='black')
    plt.quiver(0,0,w[1],w[2])
    plt.scatter(x[:,1], x[:,2], marker='o', c=y)
    plt.scatter(x[i,1], x[i,2],c='red')
    
    plt.axis([-5,10,-12,5])
    plt.show()


y = np.vectorize(lambda x: -1 if x == 1 else 1)(y)
animation_perceptron(X1,y)
#perceptron_algorithm(X1,y)
