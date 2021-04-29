import image_functions
import numpy as np


def draw_flow_intensity(op_flow: np.ndarray):
    out_im = np.zeros((op_flow.shape[0], op_flow.shape[1]))

    for row in range(0,op_flow.shape[0]):
        for col in range(0,op_flow.shape[1]):
            u = op_flow[row][col][0]
            v = op_flow[row][col][1]

            mag = np.sqrt((u*u) + (v*v))

            out_im[row][col] = mag

    return out_im

def flow_arrows(op_flow: np.ndarray):
    return