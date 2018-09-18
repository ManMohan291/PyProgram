import LDA as C


C.clearScreen()
dataTraining= C.loadData("dataTraining.txt")

X=dataTraining[:,0:2]
y=dataTraining[:,2:3]

C.plotLDA(X,y)
C.plotNormalSurface(X,y)


