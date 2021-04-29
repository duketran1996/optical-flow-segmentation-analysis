import image_functions
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plot
import cv2 as cv

def draw_color_wheel():
    image_functions.get


def draw_flow_intensity(op_flow: np.ndarray):
    out_im = np.zeros((op_flow.shape[0], op_flow.shape[1]))

    for row in range(0,op_flow.shape[0]):
        for col in range(0,op_flow.shape[1]):
            u = op_flow[row][col][0]
            v = op_flow[row][col][1]

            mag = np.sqrt((u*u) + (v*v))

            out_im[row][col] = mag

    return out_im


def draw_flow_arrows(image: np.ndarray, op_flow: np.ndarray, aperture: int, scale: int):
    '''
    op_flow = (row, col, (u,v))

    op_flow_u = u {} key = (row, col) val = u
    '''

    fig = plot.figure(figsize=(7, 7))
    plot.imshow(image, cmap='gray')
    for row in range(0, op_flow.shape[0], aperture):
        for col in range(0, op_flow.shape[1], aperture):
            plot.quiver(col, row, op_flow[row, col, 0], op_flow[row, col, 1], scale=scale, color='blue')
    plot.show()


def draw_flow_hsv(op_flow: np.ndarray) -> np.ndarray:
    hsv_output = np.zeros((op_flow.shape[0], op_flow.shape[1], 3), dtype='float32')
    hsv_output[..., 1] = 255

    magnitudes = image_functions.get_flow_magnitude_array(op_flow)
    magnitudes = image_functions.output_intensity_mapping(magnitudes)

    angles = image_functions.get_flow_angle_array(op_flow)

    for row in range(0, hsv_output.shape[0]):
        for col in range(0, hsv_output.shape[1]):
            hsv_output[row][col][0] = angles[row][col]
            hsv_output[row][col][2] = magnitudes[row][col]

    im_bgr = cv.cvtColor(hsv_output, cv.COLOR_HSV2BGR)

    return im_bgr


