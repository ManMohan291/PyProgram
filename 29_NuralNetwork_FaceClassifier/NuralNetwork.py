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
#from PIL import Image
####################################################################
def clearScreen():
    system('cls')
    return
####################################################################
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0   
    return w, b

####################################################################
def sigmoidGradient(z):
    g = 1.0 / (1.0 + np.exp(-z))
    g = g*(1-g)

    return g
####################################################################
def sigmoid(z):
    return 1/(1 + np.exp(-z))
####################################################################    
def propagate(w, b, X, Y):
  
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    
    cost = (-1.0/m)*(np.sum(Y*np.log(A)) +np.sum((1.0-Y)*np.log(1.0-A)))     
 
    dw = (1.0/m)*np.dot(X,(A-Y).T)
    db =  (1.0/m) * np.sum(A-Y)
    cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

####################################################################
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]
        
        w = w-learning_rate*dw
        b = b-learning_rate*db

        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
####################################################################
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+b) 
    for i in range(A.shape[1]):
        if(A[0,i]>0.5):
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
        
    return Y_prediction
####################################################################
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
   
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
   
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
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
def predictRunningImage(wuuuu,buuuu):
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
            if (predict(wuuuu, buuuu, X)==1):
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