import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import sys
import glob


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


X=X.reshape(X.shape[0],-1).T
Y=Y.T


# im = cv2.imread("ManMohan/frame1.jpg")
# print(type(im))

# img = Image.open( "ManMohan/frame1.jpg" )
# img.load()
# data = np.asarray( img, dtype="int32" )

print(X.shape)