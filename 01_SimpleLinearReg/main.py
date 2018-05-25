import mlib as ml
import numpy as np
ml.clearScreen()
data= ml.loadData("data.txt")
X=data[:,0:1]
y=data[:,1:2]

X=ml.addBiasVector(X)

theta =ml.initializeTheta(2)
iterations = 500
alpha = 0.0001

theta = ml.linearRegGradientDescent(X, y, theta, alpha, iterations)
Py= ml.linearRegPredict(theta,X)
ml.plotHypothesis(X[:,1],y,Py)