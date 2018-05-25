import numpy as np

def computeCost(theta,X,y):
    m,n = X.shape 
    h=np.matmul( X,theta)                      #Hypothesis
    err=h-y
    errSqr=np.multiply(err,err)
    J=(1.0/(2.0*m))* np.sum(errSqr)
    return J