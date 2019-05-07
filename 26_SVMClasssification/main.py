import SVMClassification as C

C.clearScreen()
dataTraining1= C.loadData("dataTraining1.txt")
X1=dataTraining1[:,0:2]
y1=dataTraining1[:,2:3]
dataTraining2= C.loadData("dataTraining2.txt")
X2=dataTraining2[:,0:2]
y2=dataTraining2[:,2:3]
dataTraining3= C.loadData("dataTraining3.txt")
X3=dataTraining3[:,0:2]
y3=dataTraining3[:,2:3]


plt=C.getPlot()
plt.subplot(131)
C.plotData(X1,y1)
plt.title("Dataset1")
plt.subplot(132)
C.plotData(X2,y2)
plt.title("Dataset2")
plt.subplot(133)
C.plotData(X3,y3)
plt.title("Dataset3")
plt.show()