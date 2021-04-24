import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
import math

# This is the driver function for the Horne-Schunk program.
# Users should include Horne_Schunk.py in their import list,
# after, calls can be made to the optical flow pipeline by
# calling Horne_Schunk.flow() and passing the appropriate
# arguments into the function.  Assumes the image is already
# opened as a numpy array in the calling program.
def flow(image: np.ndarray) -> np.ndarray:

    return

'''
Pipeline steps:
    Smooth image
    Sobel derivatives of x and y
    Laplacian of the image (2nd deriv)
    Gradient Magnitude on 1st derivs
    Define an aperture window
    
'''