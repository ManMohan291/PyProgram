import NuralNetwork as N


N.clearScreen()
dataTraining= N.loadData("dataTraining.txt")

X=dataTraining[:,0:400]
y=dataTraining[:,400:401]



m = X.shape[0]

rand_indices = N.getRandomValues(m)
sel = X[rand_indices[:100],:]

N.displayData(sel)




input_layer_size  = 400  
hidden_layer_size = 25   
num_labels = 10     
initial_Theta1 = N.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = N.randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = N.concatenateVectors(initial_Theta1.reshape(1,initial_Theta1.size), initial_Theta2.reshape(1,initial_Theta2.size))
initial_nn_params=initial_nn_params.flatten()

maxiter = 50
lambda_reg = 3
nn_params =N.nnOptimize(X, y,initial_nn_params,input_layer_size, hidden_layer_size, num_labels,  lambda_reg,maxiter)

yPred=N.nnPredict(nn_params,input_layer_size, hidden_layer_size, num_labels,X )


print("Accuracy="+str(N.accurracy(yPred,y)))

Theta1=nn_params[:hidden_layer_size * (input_layer_size + 1)]
Theta2=nn_params[hidden_layer_size * (input_layer_size + 1):]
Theta1.shape = (hidden_layer_size, input_layer_size + 1)
Theta2.shape =  (num_labels, hidden_layer_size + 1)

N.displayData(Theta1[:, 1:])



