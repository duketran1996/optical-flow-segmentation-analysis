import image_functions
import numpy as np
import matplotlib.pyplot as plot


def draw_flow_intensity(op_flow: np.ndarray):
    out_im = np.zeros((op_flow.shape[0], op_flow.shape[1]))

    for row in range(0,op_flow.shape[0]):
        for col in range(0,op_flow.shape[1]):
            u = op_flow[row][col][0]
            v = op_flow[row][col][1]

            mag = np.sqrt((u*u) + (v*v))

            out_im[row][col] = mag

    return out_im


def flow_arrows(image: np.ndarray, op_flow: np.ndarray):
    '''
    op_flow = (row, col, (u,v))

    op_flow_u = u {} key = (row, col) val = u
    '''

    fig = plot.figure(figsize=(7, 7))
    plot.imshow(image, cmap='gray')
    for row in range(0, op_flow.shape[0], 10):
        for col in range(0, op_flow.shape[1], 10):
            plot.quiver(col, row, op_flow[row, col, 0], op_flow[row, col, 1], scale=5, color='blue')
            print(row, col)
    plot.show()
