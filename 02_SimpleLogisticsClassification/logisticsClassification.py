from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt


####################################################################
def initTheta(size):
    return np.zeros((size, 1))
####################################################################
def listToArray(xlist):
    return np.array(xlist)
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
def sigmoid(z):
    return 1/(1 + np.exp(-z))


####################################################################
def plotDecisionBoundry(theta,X,y):
    plt.subplot(122)    
    plt.scatter(X[np.where(y==1),0],X[np.where(y==1),1],marker="+")
    plt.scatter(X[np.where(y!=1),0],X[np.where(y!=1),1],marker="o")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
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
def predict(theta,X):
    X=addBiasVector(X)
    h=np.matmul(X,theta)                      #Hypothesis
    h=sigmoid(h)
    Py=np.round(h)    
    return Py

####################################################################
def accurracy(Y1,Y2):
    m=np.mean(np.where(Y1==Y2,1,0))    
    return m*100


####################################################################
def computeCost(theta,X,y):
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
def gradientDescent(X, y, theta,alpha, iterations):
    m=len(y)
    X=addBiasVector(X)
    I=np.zeros((iterations,1),dtype=float)
    J=np.zeros((iterations,1),dtype=float)
    for k in range(iterations):
        h=np.matmul( X,theta)                      #Hypothesis
        h=sigmoid(h)
        err=h-y
        d=np.matmul(X.T,err)   #Derivative             
        I[k]=k*1.0
        J[k]=computeCost(theta,X,y)
        theta=theta -(alpha/m)*d     #Theta Itrations        
    plt.subplot(121)
    plt.plot(I, J,color='r')
    return theta
####################################################################