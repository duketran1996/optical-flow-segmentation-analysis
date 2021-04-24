import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
import math


def open_image(image_path: str):
    image = cv.imread(image_path, flags=cv.IMREAD_GRAYSCALE)
    image = ((image - np.min(image))
             * (1 / (np.max(image) - np.min(image)) * 1.0)).astype('float')
    return image


def output_intensity_mapping(image: np.ndarray):
    output_im = np.zeros(image.shape, dtype=np.uint8)
    arr_max = image.max()
    arr_min = image.min()

    for index in range(len(image.ravel())):
        output_im.ravel()[index] = \
            int(np.floor(((image.ravel()[index] - arr_min)
                          / (arr_max - arr_min)) * 255 + 0.5))

    return output_im


def output_image(image: np.ndarray, save_name):
    image_out = output_intensity_mapping(image)
    fig = plot.figure(figsize=(7, 7))
    plot.axis('off')
    plot.imshow(image_out, cmap='gray')
    plot.savefig('./results/' + save_name, bbox_inches='tight')
    plot.show()


def unpack_video(video_path: str, output_file_name: str):
    cam = cv.VideoCapture(video_path)

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print('Error: Creating director for frames')

    curr_frame = 0

    while True:
        ret, frame = cam.read()

        if ret:
            name = './data/' + output_file_name \
                   + str(curr_frame) + '.jpg'
            print('Creating...' + output_file_name
                  + str(curr_frame) + '.jpg')

            cv.imwrite(name, frame)

            curr_frame += 1
        else:
            break

    cam.release()
    cv.destroyAllWindows()


def zero_padding(num_layers: int, image: np.ndarray):
    return np.pad(image, ((num_layers, num_layers),
                          (num_layers, num_layers)),
                  'constant', constant_values=0)


def get_derivative_x(image: np.ndarray) -> np.ndarray:
    return cv.Sobel(image, ddepth=1, dx=1, dy=0, ksize=3,
                    borderType=cv.BORDER_REFLECT_101)


def get_derivative_y(image: np.ndarray) -> np.ndarray:
    return cv.Sobel(image, ddepth=1, dx=0, dy=1, ksize=3,
                    borderType=cv.BORDER_REFLECT_101)


def get_neighborhood(image: np.ndarray, row_index: int, col_index: int, ksize: int = 3):
    radius = np.floor(ksize / 2)
    col_low = col_index - radius
    col_high = col_index + radius
    row_low = row_index - radius
    row_high = row_index + radius

    neighborhood = image[row_low: row_high + 1][col_low: col_high + 1]

    return neighborhood


