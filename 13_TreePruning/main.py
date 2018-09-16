import TreeClassification as T

T.clearScreen()
dataTraining= T.loadData("dataTraining.txt")
X=dataTraining[:,0:3]
y=dataTraining[:,3:4]
Threshold=30

#Training
TrainedTree = T.SplitTree(X, y,ThresholdCount=Threshold)






#Ploting Trained Tree
plt=T.getPlot()
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')  
T.PlotTree(ax,X,y,TrainedTree)

#Ploting Pruned
TrainedTree = T.PruneTree(X, y, TrainedTree,ThresholdCount=Threshold)

newX,newY=T.PredictTree(X,y,TrainedTree)

ax = fig.add_subplot(122, projection='3d') 
T.PlotTree(ax,X,y,TrainedTree)
# plt.subplot(133) 
# plt.title("Prediction "+str(Accuracy)+"%")     
# T.PlotTree(newX,newY,TrainedTree)
plt.show()


#CheckAccuracy
Xy=T.concatenateVectors(X,y)                #Merge dataset to sort order again 
NewXy=T.concatenateVectors(newX,newY)       #Compare requires sorting as Tree shuffled the data in leaf nodes 
Accuracy=T.accurracy(Xy,NewXy)
print("Traning  accuracy(",Accuracy,"%).")

#Print Tree
T.PrintTree(TrainedTree)





