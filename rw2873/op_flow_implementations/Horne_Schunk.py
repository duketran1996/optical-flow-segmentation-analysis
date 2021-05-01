import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
import image_functions


def flow(image_1: np.ndarray, image_2: np.ndarray, lamb: float) -> np.ndarray:
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

    u = np.zeros((image_1.shape[0], image_1.shape[1]))
    v = np.zeros((image_1.shape[0], image_1.shape[1]))
    epsilon = 1e-6
    itr = 0
    max_itrs = 50
    INFO_LOSS = 1
    converged = False

    while not converged:
        old_u = u
        old_v = v

        for row in range(INFO_LOSS,u.shape[0] - INFO_LOSS):
            for col in range(INFO_LOSS,u.shape[1] - INFO_LOSS):
                u_bar = get_brightness_constancy_term(row, col, u)
                v_bar = get_brightness_constancy_term(row, col, v)

                pt_ddx = im_1_deriv_x[row][col]
                pt_ddy = im_1_deriv_y[row][col]
                pt_ddt = im_deriv_time[row][col]

                smoothness = get_smoothness_term(u_bar, v_bar, pt_ddx,
                                                 pt_ddy, pt_ddt, lamb)

                u[row][col] = u_bar - (smoothness * pt_ddx)
                v[row][col] = v_bar - (smoothness * pt_ddy)

        diff_u = np.abs(old_u - u)
        diff_v = np.abs(old_v - v)

        if (np.max(diff_u) and np.max(diff_v) < epsilon) \
                or (itr >= max_itrs):
            converged = True

        itr += 1

    optical_flow = np.zeros((image_1.shape[0], image_1.shape[1], 2))
    return optical_flow


def get_brightness_constancy_term(row, col, flow_field):
    average = 0.25 * (flow_field[row+1, col] + flow_field[row-1, col]
                      + flow_field[row, col+1] + flow_field[row, col-1])
    return average

def get_smoothness_term(ubar, vbar, im_ddx, im_ddy, im_ddt, lamb):
    numerator = im_ddx * ubar + im_ddy * vbar + im_ddt
    denom = (lamb*lamb) + (im_ddx * im_ddx) + (im_ddy * im_ddy)
    quotient = numerator / denom

    return quotient
