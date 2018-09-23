from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from scipy.stats import norm
from scipy.stats import multivariate_normal

####################################################################
def listToArray(xlist):
    return np.array(xlist)

####################################################################
def concatenateVectors(X,Y):
    r=np.column_stack((X,Y))

    return r
####################################################################
def clearScreen():
    system('cls')
    return

####################################################################
def loadData(fileName):
    data= np.loadtxt(fileName, delimiter=',')
    if (len(data.shape)==1):
        data.shape=(data.shape[0],1)
    return data


####################################################################
def PlotPoints(X,y):
    plt.scatter(X[np.where(y==1),0],X[np.where(y==1),1],marker="+")
    plt.scatter(X[np.where(y!=1),0],X[np.where(y!=1),1],marker="o")
    return
####################################################################
def getPlot():
    return plt


def plotNaiveBayes(X,y):

    cmap_light = ListedColormap(['cyan', 'orange'])
    cmap_bold = ListedColormap(['red', 'blue'])

   


    x_min, x_max = X[:, 0].min() , X[:, 0].max() 
    y_min, y_max = X[:, 1].min() , X[:, 1].max() 
    u = np.linspace(x_min, x_max,50) 
    v = np.linspace(y_min, y_max,50) 
    m=(len(u)*len(v))
    U,V=np.meshgrid(u,v)
    u=U.reshape((m,1))
    v=V.reshape((m,1))
    NewX=concatenateVectors(u,v)
     
    Newy=NaiveBayesClassifier(X,y,NewX)  



    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=y[:,0], cmap=cmap_bold)
   

    plt.subplot(122)
    plt.pcolormesh(U,V,Newy.reshape(U.shape),cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y[:,0], cmap=cmap_bold)
    plt.show()
    return
    
 
def plotNormalSurface(X,y):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
  
    M0=np.mean(X[np.where(y==0)[0]],axis=0)
    M1=np.mean(X[np.where(y==1)[0]],axis=0)
    S0=np.std(X[np.where(y==0)[0]],axis=0,ddof=1)
    S1=np.std(X[np.where(y==1)[0]],axis=0,ddof=1)
    V0=np.var(X[np.where(y==0)[0]],axis=0,ddof=1)
    V1=np.var(X[np.where(y==1)[0]],axis=0,ddof=1)
    
    x_min= M0[0]-4*S0[0]
    x_max =M0[0]+4*S0[0]
    y_min = M0[1]-4*S0[1]
    y_max = M0[1]+4*S0[1]
    u = np.linspace(x_min, x_max,50) 
    v = np.linspace(y_min, y_max,50) 
    

    U, V = np.meshgrid(u,v)
    pos = np.empty(U.shape + (2,))
    pos[:, :, 0] = U; pos[:, :, 1] = V

    rv = multivariate_normal([M0[0], M0[1]], [[V0[0], 0], [0, V0[1]]])
    W=rv.pdf(pos)

    ax.plot_surface(U,V,W,alpha=0.5, cmap='viridis',linewidth=0)



    rv = multivariate_normal([M1[0], M1[1]], [[V1[0], 0], [0, V1[1]]])
    W=rv.pdf(pos)

    ax.plot_surface(U,V,W,alpha=0.5,cmap='viridis',linewidth=0)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    
    
    plt.show()



    return


def plotNormal(X,y):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  
  
    M0=np.mean(X[np.where(y==0)[0]],axis=0)
    M1=np.mean(X[np.where(y==1)[0]],axis=0)
    S0=np.std(X[np.where(y==0)[0]],axis=0,ddof=1)
    S1=np.std(X[np.where(y==1)[0]],axis=0,ddof=1)
    V0=np.var(X[np.where(y==0)[0]],axis=0,ddof=1)
    V1=np.var(X[np.where(y==1)[0]],axis=0,ddof=1)

    #Class =0
 
   
    x_min= M0[0]-4*S0[0]
    x_max =M0[0]+4*S0[0]
    y_min = M0[1]-4*S0[1]
    y_max = M0[1]+4*S0[1]
    u = np.linspace(x_min, x_max,50) 
    v = np.linspace(y_min, y_max,50) 
    

    U, V = np.meshgrid(u,v)
    for i in range(len(v)):
        v[i]=M0[1]
    
    w = np.zeros(( len(u), len(v) ))
    U,V=np.meshgrid(u,v)
    for i in range(len(u)): 
        for j in range(len(v)):                 
            w[i,j] =norm.pdf(u[i],loc=M0[0], scale=S0[0])
    W = np.transpose(w) 

    U1=U
    V1=V
    W1=W


    
    u = np.linspace(x_min, x_max,50) 
    v = np.linspace(y_min, y_max,50) 
    w = np.linspace(x_min, x_max, 50) 

    for i in range(len(u)):
        u[i]=M0[0]
    
    w = np.zeros(( len(u), len(v) ))
    U,V=np.meshgrid(u,v)
    for i in range(len(u)): 
        for j in range(len(v)):                 
            w[i,j] =norm.pdf(v[j],loc=M0[1], scale=S0[1])
    W = np.transpose(w) 


    U2=U
    V2=V
    W2=W
    U=np.concatenate((U1,U2))
    V=np.concatenate((V1,V2))
    W=np.concatenate((W1,W2))
    #ax.plot_surface(U,V,W,alpha=0.6)
    ax.scatter(U,V,W)
    
    

   
    

 
     
    plt.show()
    return


def NaiveBayesClassifier(Xtrain,XClass,Xtest):

    M0=np.mean(Xtrain[np.where(XClass==0)[0]],axis=0)
    M1=np.mean(Xtrain[np.where(XClass==1)[0]],axis=0)
    S0=np.std(Xtrain[np.where(XClass==0)[0]],axis=0,ddof=1)
    S1=np.std(Xtrain[np.where(XClass==1)[0]],axis=0,ddof=1)
    Ytest=np.zeros((Xtest.shape[0],1))
  
    for i in range(len(Xtest[:,0:1])): 
        Prob_X0_Y0= norm.pdf(Xtest[i,0],loc=M0[0], scale=S0[0])
        Prob_X0_Y1= norm.pdf(Xtest[i,0],loc=M1[0], scale=S1[0])
        Prob_X1_Y0= norm.pdf(Xtest[i,1],loc=M0[1], scale=S0[1])
        Prob_X1_Y1= norm.pdf(Xtest[i,1],loc=M1[1], scale=S1[1])
        Prob_Y0_X1X2 =(Prob_X0_Y0 *Prob_X1_Y0)/((Prob_X0_Y0 *Prob_X1_Y0)+(Prob_X0_Y1 *Prob_X1_Y1))
        Prob_Y1_X1X2 =(Prob_X0_Y1 *Prob_X1_Y1)/((Prob_X0_Y0 *Prob_X1_Y0)+(Prob_X0_Y1 *Prob_X1_Y1))
        if (Prob_Y1_X1X2>Prob_Y0_X1X2):
            Ytest[i]=1
        
    return Ytest
