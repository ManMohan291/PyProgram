import nonLinearRegressionCurve as R
R.clearScreen()
dataTraining= R.loadData("dataTraining.txt")

X=dataTraining[:,0:1]
y=dataTraining[:,1:2]

degree=3
theta =R.initTheta(X,degree)

theta = R.optimizedGradientDescent(X, y, theta,degree)
R.plotHypothesis(theta,X,y)


dataPrediction= R.loadData("dataPrediction.txt")
PX=dataPrediction[:,0:1]
Py= R.predict(theta,PX)
print("Prediction Result:\n",R.concatenateVectors(PX,Py))
