import linearRegressionPlane as R


R.clearScreen()
dataTraining= R.loadData("dataTraining.txt")

X=dataTraining[:,0:2]
y=dataTraining[:,2:3]

degree=1

theta =R.initTheta(X,degree)
theta = R.optimizedGradientDescent(X, y, theta, degree)


#R.plotDecisionBoundry(theta,X,y)



dataPrediction= R.loadData("dataPrediction.txt")
PX=dataPrediction[:,0:2] 
Py= R.predict(theta,PX)
print("Prediction Result:\n",R.concatenateVectors(PX,Py))



