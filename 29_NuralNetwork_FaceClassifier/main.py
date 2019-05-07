import NuralNetwork as N
import numpy as np
import cv2 as cv2
import sys
import glob


import matplotlib.pyplot as plt
import h5py
import scipy
#from PIL import Image
from scipy import ndimage

N.clearScreen()

dim = 2
w, b = N.initialize_with_zeros(dim)


X,Y= N.readImageData()



d = N.model(X, Y, X, Y,  300,  0.005,  True)

# Plot learning curve (with costs)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


wuuuu=d["w"]
buuuu=d["b"]


N.predictRunningImage(wuuuu,buuuu)
