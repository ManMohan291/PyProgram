import SKLearnRegression as R

R.clearScreen()


dataTraining= R.loadData("dataTraining.txt")

X=dataTraining[:,0:1]
y=dataTraining[:,1:2]



#Plotting With Different Regularization Parameters and degree
###############################################################
plt=R.getPlot()

regAlpha=None
degree=2
RegObj=R.SKLearnRegression(X,y,degree,regAlpha,"Linear")
plt.subplot(231)
R.SKLearnPlotHypothesis(RegObj,X,y,degree,regAlpha)
predicted_y=R.SKLearnPredict(RegObj,X,degree)      
MSE1=R.SKLearnMSE(y,predicted_y)
plt.title("Linear Regression")
plt.ylabel("MSE="+str(round(MSE1,3)) )
plt.legend(("Degree="+str(degree)," Alpha="+str(regAlpha)))


regAlpha=1
degree=8
RegObj=R.SKLearnRegression(X,y,degree,regAlpha,"Ridge")
plt.subplot(232)
R.SKLearnPlotHypothesis(RegObj,X,y,degree,regAlpha)
predicted_y=R.SKLearnPredict(RegObj,X,degree)      
MSE2=R.SKLearnMSE(y,predicted_y)
plt.title("Ridge Regression")
plt.ylabel("MSE="+str(round(MSE2,3)))
plt.legend(("Degree="+str(degree)," Alpha="+str(regAlpha)))


regAlpha=None
degree=1
RegObj=R.SKLearnRegression(X,y,degree,regAlpha,"SVR")
plt.subplot(234)
R.SKLearnPlotHypothesis(RegObj,X,y,degree,regAlpha)
predicted_y=R.SKLearnPredict(RegObj,X,degree)      
MSE3=R.SKLearnMSE(y,predicted_y)
plt.xlabel("SVR Regression")
plt.ylabel("MSE="+str(round(MSE3,3))) 
plt.legend(("Degree="+str(degree)," Alpha="+str(regAlpha)))


regAlpha=None
degree=8
RegObj=R.SKLearnRegression(X,y,degree,regAlpha,"RandomForest")
plt.subplot(235)
R.SKLearnPlotHypothesis(RegObj,X,y,degree,regAlpha)
predicted_y=R.SKLearnPredict(RegObj,X,degree)      
MSE4=R.SKLearnMSE(y,predicted_y)
plt.xlabel("RandomForest Regression") 
plt.ylabel("MSE="+str(round(MSE4,3))) 
plt.legend(("Degree="+str(degree)," Alpha="+str(regAlpha)))


plt.subplot(133)
AlgoNames = ('Linear', 'Ridge', 'SVR', 'RandomForest')
AlgoIndex = [1,2,3,4]
AlgoMSE = [MSE1,MSE2,MSE3,MSE4]
plt.bar(AlgoIndex, AlgoMSE, align='center', alpha=0.5)
plt.xticks(AlgoIndex, AlgoNames)
plt.ylabel('MSE')
plt.title('Algorithm Mean Squared Error')
plt.show()
