import TreeClassification as T

T.clearScreen()
dataTraining= T.loadData("dataTraining.txt")
X=dataTraining[:,0:3]
y=dataTraining[:,3:4]
Threshold=100

#Training
TrainedTree = T.SplitTree(X, y,ThresholdCount=Threshold)
newX,newY=T.PredictTree(X,y,TrainedTree)


#CheckAccuracy
Xy=T.concatenateVectors(X,y)                #Merge dataset to sort order again 
NewXy=T.concatenateVectors(newX,newY)       #Compare requires sorting as Tree shuffled the data in leaf nodes 
Accuracy=T.accurracy(Xy,NewXy)
print("Traning  accuracy(",Accuracy,"%).")


#Ploting 
plt=T.getPlot()
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')  
T.PlotPoints(ax,X,y)
ax = fig.add_subplot(122, projection='3d') 
T.PlotTree(ax,X,y,TrainedTree)
# plt.subplot(133) 
# plt.title("Prediction "+str(Accuracy)+"%")     
# T.PlotTree(newX,newY,TrainedTree)
plt.show()

#Print Tree
T.PrintTree(TrainedTree)





