

import kmeans as K


K.clearScreen()
dataTraining= K.loadData("dataTraining.txt")

X=dataTraining[:,0:2]


initial_centroids=K.listToArray([[3, 3],[6, 2],[8,5]])

idx=K.KMean_Run(X,initial_centroids,20)

K.SaveData(K.concatenateVectors(X,idx))


K.plotKmean(X,idx)




