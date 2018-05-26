import logisticsClassification as C
C.clearScreen()
dataTraining= C.loadData("dataTraining.txt")

X=dataTraining[:,0:2]
y=dataTraining[:,2:3]

theta =C.initTheta(3)
iterations = 120
alpha = 0.001

theta = C.gradientDescent(X, y, theta, alpha, iterations)
C.plotDecisionBoundry(theta,X,y)

Py= C.predict(theta,X)
Accuracy=C.accurracy(y,Py)
print("Traning  accuracy(",Accuracy,"%).")

dataPrediction= C.loadData("dataPrediction.txt")
PX=dataPrediction[:,0:2]
Py= C.predict(theta,PX)
print("Prediction Result:\n",C.concatenateVectors(PX,Py))


