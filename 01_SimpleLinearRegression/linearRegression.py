from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

####################################################################
def initTheta(size):
    return np.zeros((size, 1))

####################################################################
def addBiasVector(X):
    return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

def concatenateVectors(X,Y):
    return np.concatenate((X,Y),axis=1)

####################################################################
def clearScreen():
    system('cls')
    return

####################################################################
def loadData(fileName):
    data= np.loadtxt(fileName, delimiter=',',unpack=True,dtype=float)
    data=data.T
    if (len(data.shape)==1):
        data.shape=(data.shape[0],1)
    return data

####################################################################
def predict(theta,X):
    X=addBiasVector(X)
    return np.matmul(X, theta)

####################################################################
def plotHypothesis(theta,X,y):
    plt.subplot(122)
    plt.scatter(X,y) 
    Py=predict(theta,X) 
    plt.plot(X, Py,color='r')
    plt.show()

####################################################################
def computeCost(theta,X,y):
    m = X.shape[0] 
    h=np.matmul( X,theta)                      #Hypothesis
    err=h-y
    errSqr=np.multiply(err,err)
    J=(1.0/(2.0*m))* np.sum(errSqr)
    return J
    
####################################################################
def gradientDescent(X, y, theta, alpha, iterations):    
    X=addBiasVector(X)
    m=len(y)
    I=np.zeros((iterations,1),dtype=float)
    J=np.zeros((iterations,1),dtype=float)
    for k in range(iterations):
        h=np.matmul( X,theta)                      #Hypothesis
        err=h-y
        d=np.matmul(err.T,X)  
        g=  alpha*((1.0/m)*d)              #Derivative
        theta=theta -g.T     #Theta Itrations        
        I[k]=k*1.0
        J[k]=computeCost(theta,X,y)
    plt.subplot(121)
    plt.plot(I, J,color='r')
    return theta


