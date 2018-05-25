import mlib as ml
ml.clearScreen()
data= ml.loadData("ex1data1.txt")

X=data[:,0]
y=data[:,1]

m=len(y)
X.shape=(m,1)
y.shape=(m,1)

X=ml.addBiasVector(X)

theta =ml.initializeTheta(2)
iterations = 500
alpha = 0.0001

theta = ml.linearRegGradientDescent(X, y, theta, alpha, iterations)
Py= ml.linearRegPredict(theta,X)
ml.plotHypothesis(X[:,1],y,Py)