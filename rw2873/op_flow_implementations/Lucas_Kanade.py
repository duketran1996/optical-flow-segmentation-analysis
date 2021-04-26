import image_functions
import numpy as np


def flow(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:
    # This is the driver function for the Lucas-Kanade program.
    # Users should include Lucas_Kanade.py in their import list,
    # after, calls can be made to the optical flow pipeline by
    # calling Lucas_Kana.flow() and passing the appropriate
    # arguments into the function.  Assumes the image is already
    # opened as a numpy array in the calling program.

    # PRECONDITION: it is assumed by the program that frame 1 is
    #               at time-step 't' and frame 2 is at time-step
    #               't+1'.

    '''
    Pipeline steps:
        Sobel derivatives of x and y
            (includes Gaussian smoothing)
        Gradient Magnitude on 1st derivs
        delta_time = frame (t) - frame(t+1)
        Brightness constancy = gm_x * u + gm_y * v + delta_time = 0
        Define an aperture window (5x5)
        Populate a linear system for each pixel in neighborhood
        Compute optical flow score
        Place into optical flow image array
    '''

    APERTURE_SIZE = 5
    INFORMATION_LIMIT = np.floor(APERTURE_SIZE / 2).astype(int)

    # Smooth and get 1st derivatives for both images (uses 3x3 Sobbel)
    im_1_deriv_x = image_functions.get_derivative_x(image_1)
    im_1_deriv_y = image_functions.get_derivative_y(image_1)

    # Calculate forward difference between the frames
    im_deriv_time = image_1 - image_2

    optical_flow = np.zeros(image_1.shape)

    for row in range(INFORMATION_LIMIT, image_1.shape[0] - INFORMATION_LIMIT):
        for col in range(INFORMATION_LIMIT, image_1.shape[1] - INFORMATION_LIMIT):
            partial_x = im_1_deriv_x[row][col]
            partial_y = im_1_deriv_y[row][col]
            deriv_time = im_deriv_time[row][col]

            # Solve for disparity parameters u and v by least squares
            neighborhood_x = image_functions.get_neighborhood(im_1_deriv_x, row, col,
                                                              APERTURE_SIZE)
            neighborhood_y = image_functions.get_neighborhood(im_1_deriv_y, row, col,
                                                              APERTURE_SIZE)
            neighborhood_t = image_functions.get_neighborhood(im_deriv_time, row, col,
                                                              APERTURE_SIZE)

            # Calculate matrix representations of derivative neighborhoods
            a = np.sum(np.dot(neighborhood_x, neighborhood_x))
            b = np.sum(np.dot(neighborhood_x, neighborhood_y))
            c = np.sum(np.dot(neighborhood_y, neighborhood_x))
            d = np.sum(np.dot(neighborhood_y, neighborhood_y))
            e = np.sum(np.dot(neighborhood_x, neighborhood_t)) * -1
            f = np.sum(np.dot(neighborhood_y, neighborhood_t)) * -1

            matrix_a = np.array([[a, b], [c, d]])
            matrix_b = np.array([[e], [f]])

            # Solve for unknown parameters with least squares
            velocities = np.linalg.lstsq(matrix_a, matrix_b)

            # Calculate optical flow value at the pixel
            optical_flow[row][col] = partial_x * velocities[0] + \
                                     partial_y * velocities[1] + \
                                     deriv_time

    return optical_flow


'''
def get_linear_system(neighborhood: list, image_deriv_x: np.ndarray,
                      image_deriv_y: np.ndarray, image_deriv_t: np.ndarray):
    a = []
    b = []

    for p in range(0,len(neighborhood)):
        a.append([image_deriv_x[neighborhood[p][0]][neighborhood[p][1]],
                 image_deriv_y[neighborhood[p][0]][neighborhood[p][1]]])
        b.append(image_deriv_t[neighborhood[p][0]][neighborhood[p][1]])

    return a, b


def solve_linear_system(a, b):
    solution = np.linalg.lstsq(a, b)
    solution = np.array(solution[0])
    return solution
'''