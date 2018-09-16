import TreeClassification as T

T.clearScreen()
dataTraining= T.loadData("dataTraining.txt")
X=dataTraining[:,0:2]
y=dataTraining[:,2:3]
Threshold=30

#Training
TrainedTree = T.SplitTree(X, y,ThresholdCount=Threshold)
newX,newY=T.PredictTree(X,y,TrainedTree)


#CheckAccuracy
Xy=T.concatenateVectors(X,y)                #Compare requires sorted order again 
NewXy=T.concatenateVectors(newX,newY)       #Compare requires sorted order again 
Accuracy=T.accurracy(Xy,NewXy)
print("Traning  accuracy(",Accuracy,"%).")


#Ploting 
plt=T.getPlot()
plt.subplot(131)  
plt.title("Dataset")  
T.PlotPoints(X,y)
plt.subplot(132)  
plt.title("Training (Threshold="+str(Threshold)+")")   
T.PlotTree(X,y,TrainedTree)
plt.subplot(133) 
plt.title("Prediction "+str(Accuracy)+"%")     
T.PlotTree(newX,newY,TrainedTree)
plt.show()

#Print Tree
T.PrintTree(TrainedTree)





