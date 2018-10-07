import SKLearnRegression as R

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
regAlphaList=[0.0,0,0,0,0,0,0,1.0,3.0,5.0,10.0,100.0]      #Lambda is named as Alpha in Ridge Regression
degreeList=[1,2,3,4,5,6,8,8,8,8,8,8]
for i in range(len(regAlphaList)):
    regAlpha=regAlphaList[i]
    degree=degreeList[i]
    RegObj=R.SKLearnRegression(Xtrain,ytrain,degree,regAlpha)
    plt.subplot(2 , int(len(regAlphaList)/2 +0.5), i+1)
    R.SKLearnPlotHypothesis(RegObj,Xtrain,ytrain,degree,regAlpha)
plt.show()


#Plotting Learning Curve
###############################################################
regAlphaList=[0,0,0.01,1]
degreeList=[1,8,8,8]
for i in range(len(regAlphaList)):
    regAlpha=regAlphaList[i]
    degree=degreeList[i]
    plt.subplot(2 , int(len(regAlphaList)/2 +0.5), i+1)
    R.plotLearningCurve(Xtrain,ytrain,Xval,yval,degree,regAlpha)
plt.show()

#Plotting Validation Curve
###############################################################
degreeList=[1,2,5,8]
regAlphaList=[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
for i in range(len(degreeList)):
    degree=degreeList[i]
    plt.subplot(2 , int(len(degreeList)/2 +0.5), i+1)
    R.plotValidationCurveForAlpha(Xtrain,ytrain,Xval,yval,degree,regAlphaList)
plt.show()



#Final Plot and Test Error
###############################################################
degree=8
regAlpha=3
plt=R.getPlot()
plt.subplot(111)
R.plotFinalCurve(Xtrain,ytrain,Xtest,ytest,degree,regAlpha)
plt.show()