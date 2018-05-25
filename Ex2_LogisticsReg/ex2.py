import mlib as ml
ml.clearScreen()
data= ml.loadData("ex2data1.txt")

X=data[:,0:2]
y=data[:,2]

m=len(y)
X.shape=(m,2)
y.shape=(m,1)

X=ml.addBiasVector(X)

theta =ml.initializeTheta(3)
iterations = 120000
alpha = 0.001

theta = ml.logisticRegGradientDescent(X, y, theta, alpha, iterations)
ml.plotDecisionBoundry(theta,X,y)

