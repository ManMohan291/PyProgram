from os import system
import numpy as np
import sklearn.linear_model  as LR
import sklearn.metrics as M
import matplotlib.pyplot as plt

####################################################################
def getPlot():
    return plt

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
def mapFeature(X,degree,includeBiasVector=True):
    
    sz=X.shape[1]
    if (sz==2):
        sz=(degree+1)*(degree+2)/2
        sz=int(sz)
    else:
         sz=degree+1

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
    if (includeBiasVector==False):
        out=out[:,1:] #Remove Bias Vector

    return out


####################################################################
def SKLearnRegression(Xtrain, ytrain,degree,regAlpha):
    Xp=mapFeature(Xtrain,degree,False)    #Polynomial  
    if (regAlpha==0):
        RegObj=LR.LinearRegression(normalize=True).fit(Xp,ytrain)
    else:
        RegObj=LR.Ridge(alpha=regAlpha,normalize=True).fit(Xp,ytrain)
    return RegObj


####################################################################
def SKLearnPredict(RegObj,X,degree):
    Xp=mapFeature(X,degree,False)    #Polynomial  
    Py=RegObj.predict(Xp)
    return Py
####################################################################

def SKLearnPlotHypothesis(RegObj,X,y,degree,regAlpha):
    plt.scatter(X,y) 
    plt.title("Alpha="+str(regAlpha)+",Degree="+str(degree))
    x_min, x_max = X[:, 0].min()-1 , X[:, 0].max()+1 
    u = np.linspace(x_min, x_max, 100)
    u.shape=(len(u),1) 
    v=SKLearnPredict(RegObj,u,degree) 
    plt.plot(u, v,color='r')
    return


####################################################################
def SKLearnMSE(y_Actual,y_Predicted):
    MSE= M.mean_squared_error(y_Actual, y_Predicted)
    return MSE

####################################################################

def plotLearningCurve(Xtrain, ytrain, Xval, yval, degree,regAlpha):
    m = len(Xtrain)
    training_error = np.zeros((m, 1))
    validation_error   = np.zeros((m, 1))
    for i in range(m):
        Current_Xtrain=Xtrain[0:i+1]
        Current_ytrain=ytrain[:i+1]
        RegObj = SKLearnRegression(Current_Xtrain, Current_ytrain,degree,regAlpha) 
        predicted_ytrain=SKLearnPredict(RegObj,Current_Xtrain,degree)       
        training_error[i]=SKLearnMSE(Current_ytrain,predicted_ytrain)
        predicted_yval=SKLearnPredict(RegObj,Xval,degree)
        validation_error[i]=SKLearnMSE(yval,predicted_yval)
    
    plt.plot(range(1,m+1), training_error)
    plt.plot( range(1,m+1), validation_error)
    plt.title('Learning Curve (Alpha = '+str(regAlpha)+',Degree='+str(degree)+')')  
    plt.legend(('Training', 'Cross Validation'))   
    plt.xlabel("Training")
    plt.ylabel("MSE")
    return

#################################################################################################################
def plotValidationCurveForAlpha(Xtrain, ytrain, Xval, yval, degree,regAlphaList):
        
    training_error = np.zeros((len(regAlphaList), 1))
    validation_error   = np.zeros((len(regAlphaList), 1))

    for i in range(len(regAlphaList)):
        regAlpha=regAlphaList[i]
        RegObj = SKLearnRegression(Xtrain,ytrain,degree,regAlpha) 

        predicted_ytrain=SKLearnPredict(RegObj,Xtrain,degree)       
        training_error[i]=SKLearnMSE(ytrain,predicted_ytrain)
        predicted_yval=SKLearnPredict(RegObj,Xval,degree)
        validation_error[i]=SKLearnMSE(yval,predicted_yval)    
    plt.plot(regAlphaList, training_error)
    plt.plot( regAlphaList, validation_error)
    plt.title('Validation Curve (Degree='+str(degree)+')')  
    plt.legend(('Training', 'Cross Validation'))   
    plt.xlabel("Alpha")
    plt.ylabel("MSE")
    return
#############################################################################################################
def plotFinalCurve(Xtrain, ytrain, Xtest, ytest, degree,regAlpha):
    RegObj = SKLearnRegression(Xtrain,ytrain,degree,regAlpha)
    predicted_ytest=SKLearnPredict(RegObj,Xtest,degree)
    testErr=SKLearnMSE(ytest,predicted_ytest)
    #PLOT   
    X=np.concatenate((Xtrain,Xtest),axis=0)
    y=np.concatenate((ytrain,ytest),axis=0)
    x_min, x_max = X[:, 0].min()-1 , X[:, 0].max()+1 
    u = np.linspace(x_min, x_max, 100)
    u.shape=(len(u),1) 
    v=SKLearnPredict(RegObj,u,degree)
    plt.plot(u, v,color='r')
    plt.scatter(Xtrain,ytrain) 
    plt.scatter(Xtest,ytest)
    plt.title("Test data Alpha="+str(regAlpha ) +" , degree="+str(degree)+" with MSE="+str(round(testErr,4)))
    plt.legend(("Regression(Alpha=3,degree=8)","Training Data","Test Data"))
    return