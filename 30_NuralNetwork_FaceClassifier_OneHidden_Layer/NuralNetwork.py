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
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed() 
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 =  np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
      
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

####################################################################
def sigmoidGradient(z):
    g = 1.0 / (1.0 + np.exp(-z))
    g = g*(1-g)
    return g
####################################################################
def sigmoid(z):
    return 1/(1 + np.exp(-z))
####################################################################
def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
####################################################################
def compute_cost(A2, Y, parameters):
    m = Y.shape[1] 
    cost = (-1.0/m)*(np.sum(Y*np.log(A2)) +np.sum((1.0-Y)*np.log(1.0-A2)))
    cost = np.squeeze(cost)                                     
    return cost
####################################################################
def backward_propagation(parameters, cache, X, Y):   
    m = X.shape[1]
 
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2-Y
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) * np.sum(dZ2,axis=1,keepdims=True)
    dZ1 =  np.dot(W2.T,dZ2)* (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1,X.T)
    db1 = (1/m) * np.sum(dZ1,axis=1,keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
####################################################################
def update_parameters(parameters, grads, learning_rate = 1.2):
   
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
####################################################################
def layer_sizes(X, Y):
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)
####################################################################
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2,0)    
    return predictions
####################################################################
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):    
    np.random.seed()
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy: {} %".format(accuracy))

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