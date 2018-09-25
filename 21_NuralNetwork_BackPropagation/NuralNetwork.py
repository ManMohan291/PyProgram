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


####################################################################
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg):
    
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F')

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F')

  
    m = len(X)             
    J = 0
    Theta1_grad = np.zeros( Theta1.shape )
    Theta2_grad = np.zeros( Theta2.shape )
    
    X = addBiasVector(X)

    a2 = sigmoid( np.matmul(X,Theta1.T) )

    
    a2 = addBiasVector(a2)

   
    a3 = sigmoid( np.matmul(a2,Theta2.T) )

    labels = y
    # set y to be matrix of size m x k
    y = np.zeros((m,num_labels))
    # for every label, convert it into vector of 0s and a 1 in the appropriate position
    for i in range(m):
    	y[i, int(labels[i])-1] = 1

    

    cost = 0
    for i in range(m):
    	cost += np.sum( y[i] * np.log( a3[i] ) + (1 - y[i]) * np.log( 1 - a3[i] ) )

    J = -(1.0/m)*cost

    #REGULARIZATION
   

    sumOfTheta1 = np.sum(np.sum(Theta1[:,1:]**2))
    sumOfTheta2 = np.sum(np.sum(Theta2[:,1:]**2))

    J = J + ( (lambda_reg/(2.0*m))*(sumOfTheta1+sumOfTheta2) )

    #BACKPROPAGATION

    bigDelta1 = 0
    bigDelta2 = 0

    # for each training example
    for t in range(m):
        x = X[t]
        a2 = sigmoid( np.matmul(x,Theta1.T) )
        a2 = np.concatenate((np.array([1]), a2))
        a3 = sigmoid( np.matmul(a2,Theta2.T) )
        delta3 = np.zeros((num_labels))
        for k in range(num_labels):
            y_k = y[t, k]
            delta3[k] = a3[k] - y_k

        delta2 = (np.matmul(Theta2[:,1:].T, delta3).T) * sigmoidGradient( np.matmul(x, Theta1.T) )

        
        bigDelta1 += np.outer(delta2, x)    # For outer product use outer instead of np.matmul(delta2[:,None], x[None,:])
        bigDelta2 += np.outer(delta3, a2)


    # step 5: obtain gradient for neural net cost function by dividing the accumulated gradients by m
    Theta1_grad = bigDelta1 / m
    Theta2_grad = bigDelta2 / m

    #% REGULARIZATION 
    Theta1_grad_unregularized = np.copy(Theta1_grad)
    Theta2_grad_unregularized = np.copy(Theta2_grad)
    Theta1_grad += (float(lambda_reg)/m)*Theta1
    Theta2_grad += (float(lambda_reg)/m)*Theta2
    Theta1_grad[:,0] = Theta1_grad_unregularized[:,0]
    Theta2_grad[:,0] = Theta2_grad_unregularized[:,0]

   
    # Unroll gradients
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))

    return J, grad
