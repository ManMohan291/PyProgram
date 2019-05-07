import SVMClassification as SVMC
from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

SVMC.clearScreen()
dataTraining= SVMC.loadData("dataTraining.txt")

X=dataTraining[:,0:2]
y=dataTraining[:,2:3]

degree=1

theta =SVMC.initTheta(X,degree)
#theta = SVMC.gradientDescent(X, y, theta,5,500, degree)
y=y.flatten()

plt.scatter(X[np.where(y==1),0],X[np.where(y==1),1],marker="+")
plt.scatter(X[np.where(y!=1),0],X[np.where(y!=1),1],marker=".")
#SVMC.plotDecisionBoundry(theta,X,y)

C=70

tol =0.001
max_passes = 20

m,n = X.shape

y[y==0] = -1

alphas = np.zeros((m, 1))
b = 0
E = np.zeros((m, 1))
passes = 0
eta = 0
L = 0
H = 0
JRANDOM=[  37,   36,    7,   18,   36,   23,   18,   13,   11,   27,    9,    1,   23,   22,   40,   31,   35,   20,    8,   40,   41,   50,   20,   21,   37,   49,   26,   40,   18,   10,   48,    3,   40,   34,   17,    4,   28,   43,    8,   18,   32,   11,    9,    5,   17,   27,   17,   13,   17,   38,   24 ]
y=y.flatten()
E=E.flatten()
alphas=alphas.flatten()

K = np.matmul(X,X.T)
while (passes < max_passes):  
    num_changed_alphas = 0
    for i in range(m):
        E[i] = b + np.sum(np.multiply(alphas,np.multiply(y,K[:,i]))) - y[i]
        if ((y[i]*E[i] < -tol and alphas[i] < C) or (y[i]*E[i] > tol and alphas[i] > 0)):
            j= np.random.randint(0,m)  #JRANDOM[i]-1
            while (i==j):
                j= np.random.randint(0,m) 
            E[j] = b + np.sum(np.multiply(alphas,np.multiply(y,K[:,j]))) - y[j]

           
            alpha_i_old = alphas[i]
            alpha_j_old = alphas[j]
            
            if (y[i] == y[j]):
                L = np.max([0, alphas[j] + alphas[i] - C])
                H = np.min([C, alphas[j] + alphas[i]])
            else:
                L =np.max([0, alphas[j] - alphas[i]])
                H = np.min([C, C + alphas[j] - alphas[i]])
            
           
            if (L == H):
                continue
            

            eta = 2.0 * K[i,j] - K[i,i] - K[j,j]
            if (eta >= 0): 
                continue
            
            
            alphas[j] = alphas[j] -(y[j] * (E[i] - E[j])) / eta
            
            
            alphas[j] = np.min ([H, alphas[j]])
            alphas[j] = np.max ([L, alphas[j]])
            
            if (np.abs(alphas[j] - alpha_j_old) < tol):
                alphas[j] = alpha_j_old
                continue
        
            
            alphas[i] = alphas[i] + y[i]*y[j]*(alpha_j_old - alphas[j])
            
            
            b1 = b - E[i] - y[i] * (alphas[i] - alpha_i_old) *  K[i,j] - y[j] * (alphas[j] - alpha_j_old) *  K[i,j]
            b2 = b - E[j] - y[i] * (alphas[i] - alpha_i_old) *  K[i,j] - y[j] * (alphas[j] - alpha_j_old) *  K[j,j]

             
            if (0 < alphas[i] and alphas[i] < C):
                b = b1
            elif (0 < alphas[j] and alphas[j] < C):
                b = b2
            else:
                b = (b1+b2)/2
            

            num_changed_alphas += 1

        #END IF
        
    #END FOR
    
    if (num_changed_alphas == 0):
        passes = passes + 1
    else:
        passes = 0

#end while
w =np.matmul(np.multiply(alphas,y).reshape(1,51),X).T
w1=w[0,0]
w2=w[1,0]


theta[0,0]=b
theta[1,0]=w1
theta[2,0]=w2

M=np.min( np.abs(b+ np.matmul(X, w))) /C  #/np.sqrt((w1**2+w2**2)))

M=1/np.sqrt((w1**2+w2**2))
SVMC.plotDecisionBoundry(theta,X,y)

theta[0,0]=b+M


SVMC.plotDecisionBoundry(theta,X,y)


theta[0,0]=b-M


SVMC.plotDecisionBoundry(theta,X,y)

plt.show()
# Py= C.predict(theta,X)
# Accuracy=C.accurracy(y,Py)
# print("Traning  accuracy(",Accuracy,"%).")

# dataPrediction= C.loadData("dataPrediction.txt")
# PX=dataPrediction[:,0:2] 
# Py= C.predict(theta,PX)
# print("Prediction Result:\n",C.concatenateVectors(PX,Py))



