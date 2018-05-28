from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

####################################################################
def initTheta(X,degree):
    size=getThetaSizeFromDegree(X,degree)
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
    degree=getDegreeFromTheta(theta,X)
    X=mapFeature(X,degree)
    Py=np.matmul(X, theta)
    return Py

####################################################################
def plotHypothesis(theta,X,y):
    plt.subplot(122)
    plt.scatter(X,y) 

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    u = np.linspace(x_min, x_max, 100)
    u.shape=(len(u),1) 
    v=predict(theta,u) 
    plt.plot(u, v,color='r')
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
def getDegreeFromTheta(theta,X):
    sz=theta.shape[0]
    if (X.shape[1]==2):
        degree=(np.sqrt(sz*8+1)-3)/2
        degree=int(degree)
    else:
         degree=sz-1
    return degree

####################################################################
def getThetaSizeFromDegree(X,degree):
    sz=X.shape[1]
    if (sz==2):
        sz=(degree+1)*(degree+2)/2
        sz=int(sz)
    else:
         sz=degree+1
    return sz

####################################################################  
def computeGradient(theta,X,y):
    m,n = X.shape
    theta.shape = (n,1) 
    h=np.matmul( X,theta)                      #Hypothesis
    err=h-y
    d=np.matmul(err.T,X)  
    g=  (1.0/m)*d
    return g.flatten()

####################################################################
def mapFeature(X,degree):
    
    sz=getThetaSizeFromDegree(X,degree)
    out=np.ones((X.shape[0],sz))

    sz=X.shape[1]
    if (sz==2):
        X1=X[:, 0:1]
        X2=X[:, 1:2]
        col=1
        for i in range(1,degree+1):        
            for j in range(0,i+1):
                out[:,col:col+1]= np.multiply(np.power(X1,i-j),np.power(X2,j))    
                col+=1
        return out
    else:
        for i in range(1,degree+1):        
            out[:,i:i+1]= np.power(X,i)
    
    return out


####################################################################
def gradientDescent(X, y, theta, alpha, iterations,degree):        
    X=mapFeature(X,degree)
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

####################################################################
def optimizedGradientDescent(X, y, theta,degree):        
    X=mapFeature(X,degree)
    Result = op.minimize(fun = computeCost, x0 = theta,  args = (X, y), method = 'TNC',jac = computeGradient)
    optimal_theta = Result.x
    return optimal_theta


