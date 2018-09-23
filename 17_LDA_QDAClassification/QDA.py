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


def plotQDA(X,y):

    cmap_light = ListedColormap(['orange', 'cyan', 'lightgreen'])
    cmap_bold = ListedColormap(['red', 'blue', 'green'])



    x_min, x_max = X[:, 0].min() , X[:, 0].max() 
    y_min, y_max = X[:, 1].min() , X[:, 1].max() 
    u = np.linspace(x_min, x_max,150) 
    v = np.linspace(y_min, y_max,150) 
    m=(len(u)*len(v))
    U,V=np.meshgrid(u,v)
    u=U.reshape((m,1))
    v=V.reshape((m,1))
    NewX=concatenateVectors(u,v)
     
    


    plt.subplot(121)
    plt.title("LDA")
    Newy=LDAClassifier(X,y,NewX)  

    plt.pcolormesh(U,V,Newy.reshape(U.shape),cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap=cmap_bold)
   

    plt.subplot(122)
    plt.title("QDA")
    Newy=QDAClassifier(X,y,NewX)  
    plt.pcolormesh(U,V,Newy.reshape(U.shape),cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap=cmap_bold)
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


def QDAClassifier(Xtrain,XClass,Xtest):


    M0=np.mean(Xtrain[np.where(XClass==0)[0]],axis=0)
    M1=np.mean(Xtrain[np.where(XClass==1)[0]],axis=0)
    M2=np.mean(Xtrain[np.where(XClass==2)[0]],axis=0)
 
    CV0=np.cov(Xtrain[np.where(XClass==0)[0]][:,0],Xtrain[np.where(XClass==0)[0]][:,1],ddof=1)
    CV1=np.cov(Xtrain[np.where(XClass==1)[0]][:,0],Xtrain[np.where(XClass==1)[0]][:,1],ddof=1)
    CV2=np.cov(Xtrain[np.where(XClass==2)[0]][:,0],Xtrain[np.where(XClass==2)[0]][:,1],ddof=1)

    PI0=(len(np.where(XClass==0)[0])/len(XClass))
    PI1=(len(np.where(XClass==1)[0])/len(XClass))
    PI2=(len(np.where(XClass==2)[0])/len(XClass))


    Mu0=[[M0[0]], [M0[1]]]
    Mu1=[[M1[0]],[M1[1]]]
    Mu2=[[M2[0]],[M2[1]]]
    

   
    D0=np.linalg.det(CV0)
    D1=np.linalg.det(CV1)
    D2=np.linalg.det(CV2)

    Z0=np.linalg.inv(CV0)
    Z1=np.linalg.inv(CV1)
    Z2=np.linalg.inv(CV2)
    #Class 0

   


   
   
  


    Ytest=np.zeros((Xtest.shape[0],1))
  
    for i in range(len(Xtest[:,0:1])): 
        X=Xtest[i]
        X=X.reshape(2,1)

        T0=np.matmul(np.matmul(np.transpose(np.subtract(X,Mu0)),Z0),np.subtract(X,Mu0))
        DS0=-0.5*T0+np.log(PI0)-0.5*np.log(D0)

        T1=np.matmul(np.matmul(np.transpose(np.subtract(X,Mu1)),Z1),np.subtract(X,Mu1))
        DS1=-0.5*T1+np.log(PI1)-0.5*np.log(D1)
        
        T2=np.matmul(np.matmul(np.transpose(np.subtract(X,Mu2)),Z2),np.subtract(X,Mu2))
        DS2=-0.5*T2+np.log(PI2)-0.5*np.log(D2)


        if (DS1>DS0 and DS1>DS2 ):
            Ytest[i]=1
        elif (DS2>DS1 and DS2>DS0):
            Ytest[i]=2   
        
        
        
    return Ytest


def LDAClassifier(Xtrain,XClass,Xtest):

    M0=np.mean(Xtrain[np.where(XClass==0)[0]],axis=0)
    M1=np.mean(Xtrain[np.where(XClass==1)[0]],axis=0)
    M2=np.mean(Xtrain[np.where(XClass==2)[0]],axis=0)
    # S0=np.std(Xtrain[np.where(XClass==0)[0]],axis=0,ddof=1)
    # S1=np.std(Xtrain[np.where(XClass==1)[0]],axis=0,ddof=1)
    # V0=np.var(Xtrain[np.where(XClass==0)[0]],axis=0,ddof=1)
    # V1=np.var(Xtrain[np.where(XClass==1)[0]],axis=0,ddof=1)
    
    CV0=np.cov(Xtrain[np.where(XClass==0)[0]][:,0],Xtrain[np.where(XClass==0)[0]][:,1],ddof=1)
    CV1=np.cov(Xtrain[np.where(XClass==1)[0]][:,0],Xtrain[np.where(XClass==1)[0]][:,1],ddof=1)
    CV2=np.cov(Xtrain[np.where(XClass==2)[0]][:,0],Xtrain[np.where(XClass==2)[0]][:,1],ddof=1)

    PI0=(len(np.where(XClass==0)[0])/len(XClass))
    PI1=(len(np.where(XClass==1)[0])/len(XClass))
    PI2=(len(np.where(XClass==2)[0])/len(XClass))

    CV0=PI0*CV0
    CV1=PI1*CV1
    CV2=PI2*CV2

    Mu0=[[M0[0]], [M0[1]]]
    Mu1=[[M1[0]],[M1[1]]]
    Mu2=[[M2[0]],[M2[1]]]

    Zigma=CV0+CV1+CV2
    Zinv=np.linalg.inv(Zigma)

    #Class 0
    DS0=np.matmul(np.matmul(Xtest,Zinv),Mu0)+(-1/2)* np.matmul(np.matmul(np.transpose(Mu0),Zinv),Mu0)+np.log(PI0)
    DS1=np.matmul(np.matmul(Xtest,Zinv),Mu1)+(-1/2)* np.matmul(np.matmul(np.transpose(Mu1),Zinv),Mu1)+np.log(PI1)
    DS2=np.matmul(np.matmul(Xtest,Zinv),Mu2)+(-1/2)* np.matmul(np.matmul(np.transpose(Mu2),Zinv),Mu2)+np.log(PI2)

    Ytest=np.zeros((Xtest.shape[0],1))
    
    for i in range(len(Xtest[:,0:1])): 
        if (DS1[i]>DS0[i] and DS1[i]>DS2[i] ):
            Ytest[i]=1
        elif (DS2[i]>DS1[i]and DS2[i]>DS0[i]):
            Ytest[i]=2   
        
    return Ytest