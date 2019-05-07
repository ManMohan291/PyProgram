from os import system
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv2
import sys
import glob
import h5py
import scipy
from scipy import ndimage
from PIL import Image
####################################################################
def clearScreen():
    system('cls')
    return

####################################################################
def initialize_parameters_deep(layer_dims):
    np.random.seed()
    parameters = {}
    L = len(layer_dims)           
    for l in range(1, L):        
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))       
    return parameters
####################################################################
def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z, cache

####################################################################
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache
####################################################################
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)],parameters["b"+str(l)], activation = "relu")
        caches+=[cache]
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)],parameters["b"+str(L)], activation = "sigmoid")
    caches+=[cache]
    return AL, caches
####################################################################
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) 
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] =linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

####################################################################
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
####################################################################
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

####################################################################
def relu(Z):
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache
####################################################################
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache
####################################################################
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m)* np.dot(dZ,A_prev.T)
    db = (1/m)* np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db
####################################################################
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
####################################################################
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]  -learning_rate*grads["dW"+ str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]  -learning_rate*grads["db"+ str(l+1)]
    return parameters

####################################################################
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1.0/m)*(np.sum(Y*np.log(AL)) +np.sum((1.0-Y)*np.log(1.0-AL)))
    cost = np.squeeze(cost)
    return cost

####################################################################
def predict(parameters,X):
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    probas, caches = L_model_forward(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
        
    return p
####################################################################
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    

    np.random.seed(1)
    costs = []                         # keep track of cost
    m=X.shape[0]
    
    parameters = initialize_parameters_deep(layers_dims)
      
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        
        grads = L_model_backward(AL, Y, caches)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    p=predict(parameters, X)
    print("Accuracy: "  + str(np.sum((p == Y)/m)))
    
    return parameters
####################################################################
def readImageData():        
    X=[]
    Y=[]
    for filename in glob.glob("ManMohan/*.jpg"):
        im = cv2.imread(filename)
        #im=im.reshape(im.shape[0],-1).T
        if(len(X)==0):
            X=[im]
            Y=[[1]]
        else:
            X=np.concatenate((X,[im]))
            Y=np.concatenate((Y,[[1]]))


    for filename in glob.glob("Pawan/*.jpg"):
        im = cv2.imread(filename)
        #im=im.reshape(im.shape[0],-1).T
        if(len(X)==0):
            X=[im]
            Y=[[0]]
        else:
            X=np.concatenate((X,[im]))
            Y=np.concatenate((Y,[[0]]))


    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X=X[s]
    Y=Y[s]

    X=X.reshape(X.shape[0],-1).T
    X=X/255
    Y=Y.T
    return X,Y

####################################################################
def predictRunningImage(parameters):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        
        )
        
    
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            count = count+1
            x0,y0=int(x),int(y)
            x1,y1=int(x+w),int(y+h)
            roi=frame[y0:y1,x0:x1]#crop 

            
        
        
            cropped=cv2.resize(roi, dsize=(150,150))
            
            
            X=cropped.reshape(1,-1).T
            X=X/255
            if (predict(parameters, X)==1):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame,"ManMohan", (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame,"Not ManMohan", (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,  0,255))
        

        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    return