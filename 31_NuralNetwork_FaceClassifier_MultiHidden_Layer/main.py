import NuralNetwork as N
import numpy as np
import cv2 as cv2
import sys
import glob


import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

N.clearScreen()



X,Y= N.readImageData()



layers_dims = [X.shape[0], 20, 7, 5, 1] #  4-layer model
parameters = N.L_layer_model(X, Y, layers_dims, num_iterations = 500, print_cost = True)



N.predictRunningImage(parameters)
