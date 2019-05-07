from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import math


####################################################################
def getRandomValues(NoOfValues):
    return np.random.permutation(NoOfValues)

####################################################################
def sigmoidGradient(z):
    g = 1.0 / (1.0 + np.exp(-z))
    g = g*(1-g)

    return g

####################################################################
def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)*(2*epsilon_init) - epsilon_init
    return W


####################################################################
def displayData(X, example_width=None):
	plt.close()
	plt.figure()
	if X.ndim == 1:
		X = np.reshape(X, (-1,X.shape[0]))
	if not example_width or not 'example_width' in locals():
		example_width = int(round(math.sqrt(X.shape[1])))

	plt.set_cmap("gray")

	m, n = X.shape
	example_height =int( n / example_width)

	display_rows = int(math.floor(math.sqrt(m)))
	display_cols = int(math.ceil(m / display_rows))

	pad = 1

	display_array = -np.ones((pad + display_rows * (example_height + pad),  pad + display_cols * (example_width + pad)))
	curr_ex = 1
	for j in range(1,display_rows+1):
		for i in range (1,display_cols+1):
			if curr_ex > m:
				break
			max_val = max(abs(X[curr_ex-1, :]))
			rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
			cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))			
			display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
			curr_ex += 1
	
		if curr_ex > m:
			break
	h = plt.imshow(display_array, vmin=-1, vmax=1)
	plt.axis('off')
	plt.show()
	return h, display_array




####################################################################
def addBiasVector(X):
    r=np.column_stack((np.ones((X.shape[0],1)),X))
    return r

def concatenateVectors(X,Y):
    r=np.column_stack((X,Y))

    return r
####################################################################
def clearScreen():
    system('cls')
    return

####################################################################
def loadData(fileName):
    data= np.loadtxt(fileName, delimiter=',')
    if (len(data.shape)==1):
        data.shape=(data.shape[0],1)
    return data

####################################################################
def sigmoid(z):
    return 1/(1 + np.exp(-z))







####################################################################
def accurracy(Y1,Y2):
    m=np.mean(Y1==Y2)   
    return m*100










####################################################################
def nnOptimize(X, y,nn_params, input_layer_size, hidden_layer_size, num_labels,lambda_reg,maxiter): 
    myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
    Result = op.minimize(fun =nnCostFunction, x0=nn_params, args=myargs, options={'disp': False, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)
    nn_params = Result.x
    return nn_params



def nnPredict(nn_params, input_layer_size, hidden_layer_size, num_labels, X):
    
    Theta1=nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Theta1 =Theta1.reshape((hidden_layer_size, input_layer_size + 1))

    Theta2=nn_params[hidden_layer_size * (input_layer_size + 1):]
    Theta2 = Theta2.reshape((num_labels, hidden_layer_size + 1))

  
    m = len(X)             
   
   
    
    # set y to be matrix of size m x k
    y = np.zeros((m,num_labels))
  
    
    #forward Propagation

    #GET Layer 1
    a1=X

    #GET Layer 2
    a1=addBiasVector(a1)
    z2=np.matmul(a1,np.transpose(Theta1))
    a2=sigmoid(z2)

    #GET LAYER 3
    a2=addBiasVector(a2)
    z3=np.matmul(a2,np.transpose(Theta2))
    a3=sigmoid(z3)

    
    #Layer3 is final Layer   
    h=a3

    y=np.argmax(h,axis=1)
    y=y+1  #index starts with 0 but label started with 1 to 10
    y=y.reshape(m,1)
    return y

####################################################################
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg):
    
    Theta1=nn_params[:hidden_layer_size * (input_layer_size + 1)]
    Theta1 =Theta1.reshape((hidden_layer_size, input_layer_size + 1))

    Theta2=nn_params[hidden_layer_size * (input_layer_size + 1):]
    Theta2 = Theta2.reshape((num_labels, hidden_layer_size + 1))

  
    m = len(X)             
    J = 0
    Theta1_grad = np.zeros( Theta1.shape )
    Theta2_grad = np.zeros( Theta2.shape )
    

    labels = y
    # set y to be matrix of size m x k
    y = np.zeros((m,num_labels))
    # for every label, convert it into vector of 0s and a 1 in the appropriate position
    for i in range(m):
    	y[i, int(labels[i])-1] = 1

    
    #forward Propagation

    #GET Layer 1
    a1=X

    #GET Layer 2
    a1=addBiasVector(a1)
    z2=np.matmul(a1,np.transpose(Theta1))
    a2=sigmoid(z2)

    #GET LAYER 3
    a2=addBiasVector(a2)
    z3=np.matmul(a2,np.transpose(Theta2))
    a3=sigmoid(z3)

    
    #Layer3 is final Layer   
    h=a3
    
    
    cost= np.subtract(-  1* np.multiply(y,np.log(h)) , np.multiply(np.subtract(np.ones(y.shape),y) , np.log(np.subtract(np.ones(h.shape),h))))
    J=1/m*np.sum(np.sum(cost))
    




    

    #BACKPROPAGATION


    #Layer 3
    err=np.subtract(h,y)
    Theta2_grad=(1/m)* np.matmul(err.T, a2)
  

    #Layer 2
    err=np.multiply(  np.matmul(err,Theta2), sigmoidGradient(addBiasVector(z2)))
    err =  err[:,1:]
    Theta1_grad=(1/m)*np.matmul(err.T, a1)
    
    #Layer 1
    #Input Layer have no error
 

    
    #% REGULARIZATION 

    regularized_Theta2=concatenateVectors(np.zeros((Theta2_grad.shape[0],1)), Theta2_grad[:,1:])
    regularized_Theta1=concatenateVectors(np.zeros((Theta1_grad.shape[0],1)), Theta1_grad[:,1:])
   
   

 
    Theta1_grad += (float(lambda_reg)/m)*regularized_Theta1
    Theta2_grad += (float(lambda_reg)/m)*regularized_Theta2
   


    J = J + ( lambda_reg*(1/(2.0*m))*(np.sum(np.sum(regularized_Theta1**2))+np.sum(np.sum(regularized_Theta2**2))) )

    # Unroll gradients
    grad = concatenateVectors(Theta1_grad.reshape(1,Theta1_grad.size), Theta2_grad.reshape(1,Theta2_grad.size))
    grad=grad.flatten()

    return J, grad
