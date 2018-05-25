import mlib as ml
ml.clearScreen()
data= ml.loadData("data.txt")

X=data[:,0:2]
y=data[:,2:3]

X=ml.addBiasVector(X)

theta =ml.initializeTheta(3)
iterations = 120000
alpha = 0.001

theta = ml.logisticRegGradientDescent(X, y, theta, alpha, iterations)
ml.plotDecisionBoundry(theta,X,y)

