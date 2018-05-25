from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
####################################################################
def initializeTheta(size):
    return np.zeros((size, 1))
####################################################################
def addBiasVector(X):
    return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
####################################################################
def clearScreen():
    system('cls')
    return
####################################################################
def loadData(fileName):
    data= np.loadtxt(fileName, delimiter=',',unpack=True,dtype=float)
    data=data.T
    return data
####################################################################
def sigmoid(z):
    return 1/(1 + np.exp(-z))
####################################################################
def linearRegPredict(theta,X):
    return np.matmul(X, theta)    
####################################################################
def plotHypothesis(X,y,Py):
    plt.subplot(122)
    plt.scatter(X,y)
    plt.plot(X, Py,color='r')
    plt.show()
####################################################################
def plotDecisionBoundry(theta,X,y):
    plt.subplot(122)    
    plt.scatter(X[np.where(y==1),1],X[np.where(y==1),2],marker="+")
    plt.scatter(X[np.where(y!=1),1],X[np.where(y!=1),2],marker="o")
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    u = np.linspace(x_min, x_max, 50) 
    v = np.linspace(y_min, y_max, 50) 
    z = np.zeros(( len(u), len(v) )) 
    for i in range(len(u)): 
        for j in range(len(v)): 
            z[i,j] = np.dot(mapFeature(np.array([u[i]]), np.array([v[j]]),1),theta) 
    z = np.transpose(z) 
    plt.contour(u, v, z, levels=[0], linewidth=2)
    plt.show()
####################################################################
def linearRegComputeCost(theta,X,y):
    m = X.shape[0] 
    h=np.matmul( X,theta)                      #Hypothesis
    err=h-y
    errSqr=np.multiply(err,err)
    J=(1.0/(2.0*m))* np.sum(errSqr)
    return J
####################################################################
def logisticRegComputeCost(theta,X,y):
    m = X.shape[0]
    h=np.matmul( X,theta)                      #Hypothesis
    h=sigmoid(h)
    term1=np.sum(np.multiply(y,np.log(h)))
    term2=np.sum(np.multiply(np.subtract(1,y),np.log(1-h)))    
    J=(-1/m)*(term1+term2)
    return J
####################################################################
def mapFeature(X1,X2,degree):
    sz=(degree+1)*(degree+2)/2
    sz=int(sz)
    out=np.ones((1,sz))
    col=1
    for i in range(1,degree+1):        
        for j in range(0,i+1):
            out[:,col]= np.multiply(np.power(X1,i-j),np.power(X2,j))    
            col+=1
    return out
####################################################################
def linearRegGradientDescent(X, y, theta, alpha, iterations):
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
        J[k]=linearRegComputeCost(theta,X,y)
    plt.subplot(121)
    plt.plot(I, J,color='r')
    return theta
####################################################################
def logisticRegGradientDescent(X, y, theta,alpha, iterations):
    m=len(y)
    I=np.zeros((iterations,1),dtype=float)
    J=np.zeros((iterations,1),dtype=float)
    for k in range(iterations):
        h=np.matmul( X,theta)                      #Hypothesis
        h=sigmoid(h)
        err=h-y
        d=np.matmul(X.T,err)   #Derivative             
        I[k]=k*1.0
        J[k]=logisticRegComputeCost(theta,X,y)
        theta=theta -(alpha/m)*d     #Theta Itrations        
    plt.subplot(121)
    plt.plot(I, J,color='r')
    return theta
####################################################################