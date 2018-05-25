import numpy as np
import computeCost as cc
import matplotlib.pyplot as plt

def gradientDescent(X, y, theta, alpha, iterations):
    m=len(y)

    #test
    I=np.zeros((iterations,1),dtype=float)
    J=np.zeros((iterations,1),dtype=float)
    for k in range(iterations):
        h=np.matmul( X,theta)                      #Hypothesis
        err=h-y
        d=np.matmul(err.T,X)  
        g=  alpha*((1.0/m)*d)              #Derivative
        theta=theta -g.T     #Theta Itrations        
        I[k]=k*1.0
        J[k]=cc.computeCost(theta,X,y)
        #New changes are here

    
    plt.subplot(121)
    plt.plot(I, J,color='r')
    

    return theta