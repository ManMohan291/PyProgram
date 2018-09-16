from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from anytree import AnyNode, RenderTree

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
        s1 = AnyNode(id="Level_"+str(Level)+"_Right("+"X"+str(SplitFeature)+">"+str(round(SplitValue,1))+")", parent=Node,vLevel=Level,vSplitFeature=SplitFeature,vOp=">",vSplitValue=SplitValue,vSplitSign=1,vPredictedClass=-1)
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
def PruneTree(X,y,Node,ThresholdCount):
    if(len(Node.children)!=0):
        SplitFeature=Node.children[0].vSplitFeature
        SplitValue=Node.children[0].vSplitValue
        X0=X[np.where(X[:,SplitFeature]<=SplitValue)]
        Y0=y[np.where(X[:,SplitFeature]<=SplitValue)]             
        X1=X[np.where(X[:,SplitFeature]>SplitValue)]
        Y1=y[np.where(X[:,SplitFeature]>SplitValue)]
        if (X0.shape[0]<ThresholdCount or X1.shape[0]<ThresholdCount):
            Node.children=[]
            PredictedClass=0
            PredictedClassLen=0
            for i in range(int(y.max()+1)):
                if (len(y[np.where(y==i)])>PredictedClassLen):
                    PredictedClass=i
                    PredictedClassLen=len(y[np.where(y==i)])
            Node.vPredictedClass=PredictedClass
        else:            
            PruneTree(X0,Y0,Node.children[0],ThresholdCount)
            PruneTree(X1,Y1,Node.children[1],ThresholdCount)
                
    return Node

####################################################################
def GetBestSplit(X,y,ThresholdCount):
    ri=0
    ci=0  
    for i in range(int(y.max()+1)):
        if(len(y[np.where(y==i)])==len(y)):
            ri=-1
            ci=-1 

    if(X.shape[0]<=ThresholdCount):
        ri=-1
        ci=-1   

    if(ri!=-1 and ci!=-1):
        G=np.zeros((X.shape))
        for ri in range(G.shape[0]):
            for ci in range(G.shape[1]):               
                G[ri,ci]=GetGiniScore(X,y,ri,ci)

        ri=np.unravel_index(np.argmax(G, axis=None), G.shape)[0]
        ci=np.unravel_index(np.argmax(G, axis=None), G.shape)[1]
    
    return ri,ci


####################################################################
def GetGiniScore(X,y,ri,ci):
    G0=0
    G1=0

    Y0=y[np.where(X[:,ci]<=X[ri,ci])]
    Y1=y[np.where(X[:,ci]>X[ri,ci])]

    if (len(Y0)!=0):
        for i in range(int(y.max()+1)):
            P=len(Y0[np.where(Y0==i)])/len(Y0)
            G0=G0+P*P

    if (len(Y1)!=0):
        for i in range(int(y.max()+1)):
            P=len(Y1[np.where(Y1==i)])/len(Y1)   
            G1=G1+P*P
    
    G_Score=(len(Y0)/len(y)) * G0 + (len(Y1)/len(y)) * G1 
    return G_Score
####################################################################
def PlotTreeSplit(ax,X,SplitFeature,SplitValue,Level): 
    x_min, x_max = X[:, 0].min() , X[:, 0].max() 
    y_min, y_max = X[:, 1].min() , X[:, 1].max()
    z_min, z_max = X[:, 2].min() , X[:, 2].max()
    u = np.linspace(x_min, x_max, 2) 
    v = np.linspace(y_min, y_max, 2)
    w = np.linspace(z_min, z_max, 2)  



    if (SplitFeature==0):
        u = np.zeros(( len(v), len(w) ))
        V,W=np.meshgrid(v,w)
        for i in range(len(v)): 
            for j in range(len(w)):                 
                u[i,j] =SplitValue
        U = np.transpose(u) 
        


    if (SplitFeature==1):
        v = np.zeros(( len(u), len(w) ))
        U,W=np.meshgrid(u,w)
        for i in range(len(u)): 
            for j in range(len(w)): 
                v[i,j] =SplitValue
        V = np.transpose(v) 
        

    if (SplitFeature==2):
        w = np.zeros(( len(u), len(v) ))
        U,V=np.meshgrid(u,v)
        for i in range(len(u)): 
            for j in range(len(v)): 
                w[i,j] =SplitValue
        W = np.transpose(w) 
    
    ax.plot_surface(U,V,W,alpha=0.6,zorder=5)
    ax.text(U[0][0], V[0][0], W[0][0], Level, color='red')

    
    
        
    return


####################################################################
def PlotTree(ax,X,y,Node):
    if(Node.id=="root"):
        ax.scatter(X[np.where(y==0),0],X[np.where(y==0),1],X[np.where(y==0),2],marker=".",facecolors='r', zorder=2)
        ax.scatter(X[np.where(y==1),0],X[np.where(y==1),1],X[np.where(y==1),2],marker=".",facecolors='g', zorder=3)
        ax.scatter(X[np.where(y==2),0],X[np.where(y==2),1],X[np.where(y==2),2],marker=".",facecolors='b', zorder=4)

    if(len(Node.children)!=0):
        SplitFeature=Node.children[0].vSplitFeature
        SplitValue=Node.children[0].vSplitValue
        Level=Node.children[0].vLevel
        X0=X[np.where(X[:,SplitFeature]<=SplitValue)]
        Y0=y[np.where(X[:,SplitFeature]<=SplitValue)]     
        X1=X[np.where(X[:,SplitFeature]>SplitValue)]
        Y1=y[np.where(X[:,SplitFeature]>SplitValue)]
        PlotTreeSplit(ax,X,SplitFeature,SplitValue,Level)
        PlotTree(ax,X0,Y0,Node.children[0])
        PlotTree(ax,X1,Y1,Node.children[1])
    
    return

####################################################################
def PlotPoints(ax,X,y):

    
    ax.scatter(X[np.where(y==0),0],X[np.where(y==0),1],X[np.where(y==0),2],marker="o",facecolors='r')
    ax.scatter(X[np.where(y==1),0],X[np.where(y==1),1],X[np.where(y==1),2],marker="o",facecolors='g')
    ax.scatter(X[np.where(y==2),0],X[np.where(y==2),1],X[np.where(y==2),2],marker="o",facecolors='b')
    return

####################################################################
def PrintTree(Tree):
    print(RenderTree(Tree))
    return
