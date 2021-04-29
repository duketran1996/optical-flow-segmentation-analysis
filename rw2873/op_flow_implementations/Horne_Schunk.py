import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
import image_functions


def flow(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:
    # This is the driver function for the Horne-Schunk program.
    # Users should include Horne_Schunk.py in their import list,
    # after, calls can be made to the optical flow pipeline by
    # calling Horne_Schunk.flow() and passing the appropriate
    # arguments into the function.  Assumes the image is already
    # opened as a numpy array in the calling program.

    # Smooth and get 1st derivatives for both images (uses 3x3 Sobbel)
    im_1_deriv_x = image_functions.get_derivative_x(image_1)
    im_1_deriv_y = image_functions.get_derivative_y(image_1)

    # Calculate forward difference between the frames
    im_deriv_time = image_1 - image_2

    # Initialize output array and looping parameters
    optical_flow = np.zeros((image_1.shape[0], image_1.shape[1], 2))
    converged = False

    while not converged:
        for rows in range(0,optical_flow.shape[0]):
            for col in range(0,optical_flow.shape[1]):

                ubar = 0.25 * ()

    return optical_flow

'''
Pipeline steps:
    Smooth image
    Sobel derivatives of x and y
    Laplacian of the image (2nd deriv)
    Gradient Magnitude on 1st derivs
    Define an aperture window
    
'''