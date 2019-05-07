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



H=4

parameters = N.nn_model(X, Y, H, num_iterations = 100, print_cost=True)




N.predictRunningImage(parameters)
