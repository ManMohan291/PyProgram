import nonLinearRegressionCurve as R

R.clearScreen()
dataTraining= R.loadData("dataTraining.txt")
dataTest= R.loadData("dataTest.txt")
dataValidation= R.loadData("dataValidation.txt")
Xtrain=dataTraining[:,0:1]
ytrain=dataTraining[:,1:2]
Xtest=dataTest[:,0:1]
ytest=dataTest[:,1:2]
Xval=dataValidation[:,0:1]
yval=dataValidation[:,1:2]



#Plotting With Different Regularization Parameters and degree
###############################################################
plt=R.getPlot()
regLambdaList=[0,0,0,0,0,0,0,1,3,5,10,100]
degreeList=[1,2,3,4,5,6,8,8,8,8,8,8]
for i in range(len(regLambdaList)):
    regLambda=regLambdaList[i]
    degree=degreeList[i]
    Xp=R.mapFeature(Xtrain,degree)    #Polynomial
    Xn, mu, sigma = R.featureNormalize(Xp)  # Normalize
    theta = R.optimizedGradientDescent(Xn, ytrain, degree,regLambda)  #Without Lib   theta = R.gradientDescent(Xn, y, theta,alpha,iter,degree,regLambda)
    plt.subplot(2 , int(len(regLambdaList)/2 +0.5), i+1)
    R.plotHypothesis(theta,Xtrain,ytrain,regLambda,mu, sigma)
plt.show()


#Plotting Learning Curve
###############################################################
regLambdaList=[0,0,0.01,1]
degreeList=[1,8,8,8]
for i in range(len(regLambdaList)):
    regLambda=regLambdaList[i]
    degree=degreeList[i]
    plt.subplot(2 , int(len(regLambdaList)/2 +0.5), i+1)
    R.plotLearningCurve(Xtrain,ytrain,Xval,yval,degree,regLambda)
plt.show()

#Plotting Validation Curve
###############################################################
degreeList=[1,2,5,8]
regLambdaList=[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
for i in range(len(degreeList)):
    degree=degreeList[i]
    plt.subplot(2 , int(len(degreeList)/2 +0.5), i+1)
    R.plotValidationCurveForLambda(Xtrain,ytrain,Xval,yval,degree,regLambdaList)
plt.show()



#Final Plot and Test Error
###############################################################
degree=8
regLambda=3
plt=R.getPlot()
plt.subplot(111)
R.plotFinalCurve(Xtrain,ytrain,Xtest,ytest,degree,regLambda)
plt.show()