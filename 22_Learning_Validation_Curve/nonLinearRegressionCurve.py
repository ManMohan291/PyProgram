from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

####################################################################
def getPlot():
    return plt

####################################################################
def initTheta(X,degree):
    size=getThetaSizeFromDegree(X,degree)
    #newTheta=np.random.rand(size, 1)*1000
    newTheta=np.zeros((size, 1))
    return  newTheta

####################################################################
def addBiasVector(X):
    return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

####################################################################
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
def predict(theta,X,mu, sigma):
    degree=getDegreeFromTheta(theta,X)
    X=mapFeature(X,degree)
    X = X - mu
    X = X/sigma
    Py=np.matmul(X, theta)
    return Py

####################################################################
def plotHypothesis(theta,X,y,regLambda,mu, sigma):
    degree=getDegreeFromTheta(theta,X)
    plt.scatter(X,y) 
    plt.title("Lambda="+str(regLambda)+",Degree="+str(degree))
    x_min, x_max = X[:, 0].min()-1 , X[:, 0].max()+1 
    u = np.linspace(x_min, x_max, 100)
    u.shape=(len(u),1) 
    v=predict(theta,u,mu, sigma) 
    plt.plot(u, v,color='r')
    


####################################################################
def computeCost(theta,X,y,regLambda):
    m,n = X.shape
    theta.shape = (n,1)

    h=np.matmul( X,theta)                      #Hypothesis
    err=h-y
    errSqr=err**2
    J=(1.0/(2.0*m))* np.sum(errSqr)
    
    
    regularized_theta=np.concatenate((np.zeros((1,1)) , theta[1:,:]),axis=0)
    J=J +regLambda* (1.0/(2.0*m)) *(np.sum(regularized_theta**2))
 
    
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
def computeGradient(theta,X,y,regLambda):
    m,n = X.shape
    theta.shape = (n,1) 
    h=np.matmul( X,theta)                      #Hypothesis
    err=h-y
    d=np.matmul(err.T,X)  
    g=  (1.0/m)*d
    g=g.T
    regularized_theta=np.concatenate((np.zeros((1,1)) , theta[1:,:]),axis=0)
        
    g=g + (regLambda/m)*regularized_theta

    

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
def gradientDescent(X, y, theta, alpha, iterations,degree,regLambda):        
    m=len(y)
    I=np.zeros((iterations,1),dtype=float)
    J=np.zeros((iterations,1),dtype=float)
    for k in range(iterations):
        h=np.matmul( X,theta)                      #Hypothesis
        err=h-y
        d=np.matmul(err.T,X)  
        g=  alpha*((1.0/m)*d)              #Derivative
        g=g.T     #Theta Itrations  
        regularized_theta=np.concatenate((np.zeros((1,1)) , theta[1:,:]),axis=0)
        g=g+ (regLambda/m)*regularized_theta 
        I[k]=k*1.0
        J[k]=computeCost(theta,X,y,regLambda)
        theta=theta-g
    
    
    
    plt.plot(I, J,color='r')
    return theta

####################################################################
def optimizedGradientDescent(X, y,degree,regLambda):        
    maxiter = 200
    theta =np.zeros((X.shape[1], 1))
    Result = op.minimize(fun = computeCost, x0 = theta,  args = (X, y,regLambda), options={'disp': False, 'maxiter':maxiter}, method="L-BFGS-B",jac = computeGradient)
    optimal_theta = Result.x
    return optimal_theta
####################################################################

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    #Handle Bias Vector
    if(mu[0]==1 and sigma[0]==0):
        mu[0]=0
        sigma[0]=1
    X_norm = X - mu
    X_norm = X_norm/sigma
    return X_norm, mu, sigma
####################################################################

def plotLearningCurve(Xtrain, ytrain, Xval, yval, degree,regLambda):
    m = len(Xtrain)
    Xtrain_p=mapFeature(Xtrain,degree)    #Polynomial
    Xtrain_n, mu, sigma = featureNormalize(Xtrain_p)  # Normalize

    Xval_p=mapFeature(Xval,degree)    #Polynomial
    Xval_n = Xval_p - mu
    Xval_n = Xval_n/sigma
    
    training_error = np.zeros((m, 1))
    validation_error   = np.zeros((m, 1))

    for i in range(m):
        #Learning
        Current_Xtrain=Xtrain_n[0:i+1]
        Current_ytrain=ytrain[:i+1]
        theta = optimizedGradientDescent(Current_Xtrain, Current_ytrain,degree,regLambda)        
        training_error[i]=computeCost(theta,Current_Xtrain,Current_ytrain,0)
        validation_error[i]=computeCost(theta,Xval_n,yval,0)
    
    plt.plot(range(1,m+1), training_error)
    plt.plot( range(1,m+1), validation_error)
    plt.title('Learning Curve (Lambda = '+str(regLambda)+',Degree='+str(degree)+')')  
    plt.legend(('Training', 'Cross Validation'))   
    plt.xlabel("Training")
    plt.ylabel("Error/Cost")
    return

#################################################################################################################
def plotValidationCurveForLambda(Xtrain, ytrain, Xval, yval, degree,regLambdaList):
    
    Xtrain_p=mapFeature(Xtrain,degree)    #Polynomial
    Xtrain_n, mu, sigma = featureNormalize(Xtrain_p)  # Normalize

    Xval_p=mapFeature(Xval,degree)    #Polynomial
    Xval_n = Xval_p - mu
    Xval_n = Xval_n/sigma
    
    training_error = np.zeros((len(regLambdaList), 1))
    validation_error   = np.zeros((len(regLambdaList), 1))

    for i in range(len(regLambdaList)):
        regLambda=regLambdaList[i]
        theta = optimizedGradientDescent(Xtrain_n, ytrain, degree,regLambda)
        training_error[i]=computeCost(theta,Xtrain_n,ytrain,0)
        validation_error[i]=computeCost(theta,Xval_n,yval,0)
    
    plt.plot(regLambdaList, training_error)
    plt.plot( regLambdaList, validation_error)
    plt.title('Validation Curve (Degree='+str(degree)+')')  
    plt.legend(('Training', 'Cross Validation'))   
    plt.xlabel("Lambda")
    plt.ylabel("Error/Cost")
    return
#############################################################################################################
def plotFinalCurve(Xtrain, ytrain, Xtest, ytest, degree,regLambda):
    Xp=mapFeature(Xtrain,degree)    #Polynomial
    Xn, mu, sigma = featureNormalize(Xp)  # Normalize
    theta = optimizedGradientDescent(Xn, ytrain, degree,regLambda)        
    Xtest_p=mapFeature(Xtest,degree)    #Polynomial
    Xtest_n = Xtest_p - mu
    Xtest_n = Xtest_n/sigma
    testErr=computeCost(theta,Xtest_n,ytest,0)
    #PLOT   
    X=np.concatenate((Xtrain,Xtest),axis=0)
    y=np.concatenate((ytrain,ytest),axis=0)
    x_min, x_max = X[:, 0].min()-1 , X[:, 0].max()+1 
    u = np.linspace(x_min, x_max, 100)
    u.shape=(len(u),1) 
    v=predict(theta,u,mu, sigma) 
    plt.plot(u, v,color='r')
    plt.scatter(Xtrain,ytrain) 
    plt.scatter(Xtest,ytest)
    plt.title("Test data Lambda="+str(regLambda ) +" , degree="+str(degree)+" with Error="+str(round(testErr,4)))
    plt.legend(("Regression(Lambda=3,degree=8)","Training Data","Test Data"))
    return