from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from anytree import AnyNode, RenderTree

####################################################################
def initTheta(size):
    return np.zeros((size, 1))
####################################################################
def listToArray(xlist):
    return np.array(xlist)
####################################################################
def addBiasVector(X):
    return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
####################################################################
def concatenateVectors(X,Y):
    return np.concatenate((X,Y),axis=1)
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
def accurracy(Xy,NewXy):
    Xy=np.sort(Xy,axis=0)
    NewXy=np.sort(NewXy,axis=0)
    Y1=Xy[:,-1]
    Y2=NewXy[:,-1]
    m=np.mean(np.where(Y1==Y2,1,0))    
    return m*100

####################################################################
def SplitTree(X, y,Level=1,Node=AnyNode(id="root",vPredictedClass=-1),ThresholdCount=1):
     
    ri,ci=GetBestSplit(X,y,ThresholdCount)
  
    if( ri!=-1 and     ci!=-1):
        SplitFeature=ci
        SplitValue=X[ri,ci]

        #PlotTreeSplit(X,SplitFeature,SplitValue,Level)  #Plot While Training
        
        X0=X[np.where(X[:,SplitFeature]<=SplitValue)]
        Y0=y[np.where(X[:,SplitFeature]<=SplitValue)]     
       
        X1=X[np.where(X[:,SplitFeature]>SplitValue)]
        Y1=y[np.where(X[:,SplitFeature]>SplitValue)]
       

        s0 = AnyNode(id="Level_"+str(Level)+"_Left("+"X"+str(SplitFeature)+"<"+str(round(SplitValue,1))+")", parent=Node,vLevel=Level,vSplitFeature=SplitFeature,vOp="<",vSplitValue=SplitValue,vSplitSign=-1,vPredictedClass=-1)
        s1 = AnyNode(id="Level_"+str(Level)+"_Right("+"X"+str(SplitFeature)+"<"+str(round(SplitValue,1))+")", parent=Node,vLevel=Level,vSplitFeature=SplitFeature,vOp=">",vSplitValue=SplitValue,vSplitSign=1,vPredictedClass=-1)
        s0=SplitTree(X0,Y0,Level+1,s0,ThresholdCount=ThresholdCount)        
        s1=SplitTree(X1,Y1,Level+1,s1,ThresholdCount=ThresholdCount)

    else:
        if len(y[np.where(y==0)])<= len(y[np.where(y==1)]):
            Node.vPredictedClass=1
        else:
            Node.vPredictedClass=0
      

    return Node
####################################################################
def PredictTree(X,y,Node):
    if(len(Node.children)!=0):
        SplitFeature=Node.children[0].vSplitFeature
        SplitValue=Node.children[0].vSplitValue
        X0=X[np.where(X[:,SplitFeature]<=SplitValue)]
        Y0=y[np.where(X[:,SplitFeature]<=SplitValue)]             
        X1=X[np.where(X[:,SplitFeature]>SplitValue)]
        Y1=y[np.where(X[:,SplitFeature]>SplitValue)]
        newX1,newY1=PredictTree(X0,Y0,Node.children[0])
        newX2,newY2=PredictTree(X1,Y1,Node.children[1])
        newX= np.concatenate((newX1,newX2),axis=0)
        newY=np.concatenate((newY1,newY2),axis=0)
    else:
        newX=X
        for i in range(len(y)):
            y[i]=Node.vPredictedClass
        newY=y
    return newX,newY

####################################################################
def GetBestSplit(X,y,ThresholdCount):
    if(X.shape[0]<=ThresholdCount or len(y[np.where(y==0)])==0 or len(y[np.where(y==1)])==0):
        ri=-1
        ci=-1        
    else:
        ri=0
        ci=0
        G=np.zeros((X.shape))
        for ri in range(G.shape[0]):
            for ci in range(G.shape[1]):               
                G[ri,ci]=GetGiniScore(X,y,ri,ci)

        ri=np.unravel_index(np.argmax(G, axis=None), G.shape)[0]
        ci=np.unravel_index(np.argmax(G, axis=None), G.shape)[1]
    return ri,ci

####################################################################
def GetGiniScore(X,y,ri,ci):
    P0F=0
    P0S=0
    P1F=0
    P1S=0
    Y0=y[np.where(X[:,ci]<=X[ri,ci])]
    if (len(Y0)!=0):
        P0F=len(Y0[np.where(Y0==0)])/len(Y0)
        P0S=len(Y0[np.where(Y0==1)])/len(Y0)

    G0=P0S*P0S+P0F*P0F

    Y1=y[np.where(X[:,ci]>X[ri,ci])]
    if (len(Y1)!=0):
        P1F=len(Y1[np.where(Y1==0)])/len(Y1)
        P1S=len(Y1[np.where(Y1==1)])/len(Y1)

    G1=P1S*P1S+P1F*P1F
    
    G_Score=(len(Y0)/len(y)) * G0 + (len(Y1)/len(y)) * G1 
    return G_Score

####################################################################
def PlotTreeSplit(X,SplitFeature,SplitValue,Level): 
    x_min, x_max = X[:, 0].min() , X[:, 0].max() 
    y_min, y_max = X[:, 1].min() , X[:, 1].max()
    u = np.linspace(x_min, x_max, 2) 
    v = np.linspace(y_min, y_max, 2)      
    for i in range(len(v)): 
        if (SplitFeature==0):        
            u[i] = SplitValue
        else:
            v[i] = SplitValue
    plt.plot(u, v)
    plt.text(u[0],v[0],Level,rotation=90*SplitFeature )
    return


####################################################################
def PlotTree(X,y,Node):
    if(len(Node.children)!=0):
        SplitFeature=Node.children[0].vSplitFeature
        SplitValue=Node.children[0].vSplitValue
        Level=Node.children[0].vLevel
        X0=X[np.where(X[:,SplitFeature]<=SplitValue)]
        Y0=y[np.where(X[:,SplitFeature]<=SplitValue)]     
        X1=X[np.where(X[:,SplitFeature]>SplitValue)]
        Y1=y[np.where(X[:,SplitFeature]>SplitValue)]
        PlotTreeSplit(X,SplitFeature,SplitValue,Level)
        PlotTree(X0,Y0,Node.children[0])
        PlotTree(X1,Y1,Node.children[1])
    else:
        plt.scatter(X[np.where(y==1),0],X[np.where(y==1),1],marker="+")
        plt.scatter(X[np.where(y!=1),0],X[np.where(y!=1),1],marker="o")
    return

####################################################################
def PlotPoints(X,y):
    plt.scatter(X[np.where(y==1),0],X[np.where(y==1),1],marker="+")
    plt.scatter(X[np.where(y!=1),0],X[np.where(y!=1),1],marker="o")
    return

####################################################################
def PrintTree(Tree):
    print(RenderTree(Tree))
    return
