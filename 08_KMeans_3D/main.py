

import kmeans as K


K.clearScreen()
dataTraining= K.loadData("dataTraining.txt")

X=dataTraining[:,0:3]


initial_centroids=K.listToArray([[3, 3,3],[6, 2,4],[8,5,7]])

idx=K.KMean_Run(X,initial_centroids,100)




K.plotKmean(X,idx)




