import linearRegressionPlane as R


R.clearScreen()
dataTraining= R.loadData("dataTraining.txt")




X1=dataTraining[:,0:1]
X2=dataTraining[:,1:2]

degree=2


theta1 =R.initTheta(X1,degree)
theta1 = R.optimizedGradientDescent(X1, X2, theta1, degree)




X=dataTraining[:,0:2]
y=dataTraining[:,2:3]


theta =R.initTheta(X,degree)
theta = R.optimizedGradientDescent(X, y, theta, degree)


R.plotLine3d(theta1,theta,X,y)




