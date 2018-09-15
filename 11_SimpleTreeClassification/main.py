import TreeClassification as T




T.clearScreen()
dataTraining= T.loadData("dataTraining.txt")

X=dataTraining[:,0:2]
y=dataTraining[:,2:3]


#Training
TrainedTree = T.SplitTree(X, y,ThresholdCount=30)
newX,newY=T.PredictTree(X,y,TrainedTree)



#Ploting 
plt=T.getPlot()

plt.subplot(131)    
T.PlotPoints(X,y)

plt.subplot(132)    
T.PlotTree(X,y,TrainedTree)

plt.subplot(133)    
T.PlotTree(newX,newY,TrainedTree)

plt.show()



#Print Tree
T.PrintTree(TrainedTree)


#CheckAccuracy

Xy=T.concatenateVectors(X,y)                #Compare require sorted order again 
NewXy=T.concatenateVectors(newX,newY)       #Compare require sorted order again 

Accuracy=T.accurracy(Xy,NewXy)
print("Traning  accuracy(",Accuracy,"%).")




