import linearRegression as R
R.clearScreen()
dataTraining= R.loadData("dataTraining.txt")

X=dataTraining[:,0:1]
y=dataTraining[:,1:2]

theta =R.initTheta(2)
iterations = 500
alpha = 0.0001

theta = R.gradientDescent(X, y, theta, alpha, iterations)
R.plotHypothesis(theta,X,y)


dataPrediction= R.loadData("dataPrediction.txt")
PX=dataPrediction[:,0:1]
Py= R.predict(theta,PX)
print("Prediction Result:\n",R.concatenateVectors(PX,Py))
