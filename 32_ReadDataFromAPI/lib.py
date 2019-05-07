from os import system
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import math



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
def getRandomValues(NoOfValues):
    return np.random.permutation(NoOfValues)

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
def initTheta(X,degree):
    size=getThetaSizeFromDegree(X,degree)
    return np.zeros((size, 1))
####################################################################
def listToArray(xlist):
    return np.array(xlist)

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
def plotDecisionBoundry(theta,X,y):
    degree=getDegreeFromTheta(theta,X)

    #plt.subplot(122)    
    plt.scatter(X[np.where(y==1),0],X[np.where(y==1),1],marker="+")
    plt.scatter(X[np.where(y!=1),0],X[np.where(y!=1),1],marker="o")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    u = np.linspace(x_min, x_max, 10) 
    v = np.linspace(y_min, y_max, 10) 
    #U,V=np.meshgrid(u,v)
    z = np.zeros(( len(u), len(v) )) 
    for i in range(len(u)): 
        for j in range(len(v)): 
            uv= concatenateVectors(np.array([[u[i]]]),np.array([[v[j]]]))
            z[i,j] =np.sum( np.matmul(mapFeature(uv,degree),theta) )
    z = np.transpose(z) 
    plt.contour(u, v, z, levels=[0], linewidth=2)
    plt.show()

####################################################################
def getDegreeFromTheta(theta,X):
    sz=theta.shape[0]
    if (X.shape[1]==2):
        degree=(np.sqrt(sz*8+1)-3)/2
        degree=int(degree)
    else:
         degree=sz-1
    return degree

####################################################################
def getThetaSizeFromDegree(X,degree):
    sz=X.shape[1]
    if (sz==2):
        sz=(degree+1)*(degree+2)/2
        sz=int(sz)
    else:
         sz=degree+1
    return sz

####################################################################
def predict(theta,X):
    degree=getDegreeFromTheta(theta,X)
    X=mapFeature(X,degree)
    h=np.matmul(X,theta)                      #Hypothesis
    h=sigmoid(h)
    Py=(h>=0.5)*1    
    return Py

####################################################################
def accurracy(Y1,Y2):
    m=np.mean(Y1==Y2)   
    return m*100


####################################################################
def computeCost(theta,X,y):
    m = X.shape[0]
    h= X @ theta                      #Hypothesis
    h=sigmoid(h)
    h.shape=y.shape
    term1= y *  np.log(h) 
    term2= (1-y) * np.log(1-h)    
    J=(-1/m)*(term1+term2).sum()
    return J

####################################################################
def mapFeature(X,degree):
    
    sz=getThetaSizeFromDegree(X,degree)
    out=np.ones((X.shape[0],sz))

    sz=X.shape[1]
    if (sz==2):
        X1=X[:, 0:1]
        X2=X[:, 1:2]
        col=1
        for i in range(1,degree+1):        
            for j in range(0,i+1):
                out[:,col:col+1]= np.multiply(np.power(X1,i-j),np.power(X2,j))    
                col+=1
        return out
    else:
        for i in range(1,degree+1):        
            out[:,i:i+1]= np.power(X,i)
    
    return out


####################################################################
def gradientDescent(X, y, theta,alpha, iterations,degree):
    m=len(y)
    X=mapFeature(X,degree)
    I=np.zeros((iterations,1),dtype=float)
    J=np.zeros((iterations,1),dtype=float)
    for k in range(iterations):
        h=np.matmul( X,theta)                      #Hypothesis
        h=sigmoid(h)
        err=h-y
        d=np.matmul(X.T,err)   #Derivative             
        I[k]=k*1.0
        J[k]=0
        J[k]=computeCost(theta,X,y)
        
        
        theta=theta -(alpha/m)*d     #Theta Itrations        
    plt.subplot(121)
    plt.plot(I, J)
    return theta

####################################################################
def computeGradient(theta,X,y):
    m,n = X.shape
    theta.shape = (n,1) 
    h=np.matmul( X,theta)                      #Hypothesis
    h=sigmoid(h)
    h.shape=y.shape
    err=h-y
    d=np.matmul(err.T,X)  
    g=  (1.0/m)*d
    return g.flatten()




####################################################################
def optimizedGradientDescent(X, y, theta,degree): 
    oldShape=theta.shape
    X=mapFeature(X,degree)
    myargs=(X, y[:,0])
    Result = op.minimize(fun = computeCost, x0 = theta.flatten(),  args =myargs, method = 'TNC',jac = computeGradient)
    theta = Result.x
    
   
    #theta = op.fmin(computeCost, x0=theta, args=myargs) 
    #theta,_,_,_,_,_,_= op.fmin_bfgs(computeCost, x0=theta, args=myargs, full_output=True) 
    theta.shape=oldShape

    return theta



####################################################################
def NuralMinimize(X, y,nn_params, input_layer_size, hidden_layer_size, num_labels,lambda_reg,maxiter): 
    myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
    Result = op.minimize(fun =nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)
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
