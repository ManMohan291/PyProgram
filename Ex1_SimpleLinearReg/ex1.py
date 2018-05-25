from os import system
import matplotlib.pyplot as plt
import numpy as np
import gradientDescent as gd
#from gradientDescent import gradientDescent
system('cls')
data= np.loadtxt("ex1data1.txt", delimiter=',',unpack=True,dtype=float)
plt.figure(1)

X=data.T[:,0]
y=data.T[:,1]

m=len(y)
X.shape=(m,1)
y.shape=(m,1)





X=np.concatenate((np.ones((m,1)),X),axis=1)
theta = np.zeros((2, 1))


iterations = 500
alpha = 0.0001

theta = gd.gradientDescent(X, y, theta, alpha, iterations)

yn= np.matmul(X, theta)
plt.subplot(122)
plt.scatter(X[:,1],y)
plt.plot(X[:,1], yn,color='r')
plt.show()