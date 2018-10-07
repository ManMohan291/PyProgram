from os import system
import numpy as np
import sklearn.linear_model  as LR
import sklearn.svm as SM
import sklearn.ensemble as RF
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
def SKLearnRegression(Xtrain, ytrain,degree,regAlpha,algorithm):
    Xp=mapFeature(Xtrain,degree,False)    #Polynomial  
    if (algorithm=="Linear"):
        RegObj=LR.LinearRegression(normalize=True).fit(Xp,ytrain)
    elif (algorithm=="Ridge"):
        RegObj=LR.Ridge(alpha=regAlpha,normalize=True).fit(Xp,ytrain)
    elif (algorithm=="SVR"):
        RegObj=SM.SVR(degree=degree).fit(Xp,ytrain)
    elif (algorithm=="RandomForest"):
        RegObj=RF.RandomForestRegressor().fit(Xp,ytrain)
    else:
        RegObj=LR.LinearRegression(normalize=True).fit(Xp,ytrain)
    return RegObj


####################################################################
def SKLearnPredict(RegObj,X,degree):
    Xp=mapFeature(X,degree,False)    #Polynomial  
    Py=RegObj.predict(Xp)
    return Py
####################################################################

def SKLearnPlotHypothesis(RegObj,X,y,degree,regAlpha):
    plt.scatter(X,y) 
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

