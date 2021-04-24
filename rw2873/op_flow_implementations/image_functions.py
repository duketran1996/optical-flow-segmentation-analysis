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


def get_operation_order(pixel_coord: tuple, filter_size: tuple, operation_type: bool):
    index_pairs: list = []
    low_x = -1 * int(np.floor(filter_size[1] / 2))
    high_x = int(np.floor(filter_size[1] / 2))
    low_y = -1 * int(np.floor(filter_size[0] / 2))
    high_y = int(np.floor(filter_size[0] / 2))

    if filter_size == (1, 1):
        index_pairs.append(((pixel_coord), (0, 0)))
        return index_pairs

    # Cross-Correlation
    if operation_type:
        for col in range(low_x, high_x + 1, 1):
            for row in range(low_y, high_y + 1, 1):
                index_pairs.append(((pixel_coord[0] + row, pixel_coord[1] + col),
                                    (row + high_y, col + high_x)))

    # Convolution
    else:
        for col in range(high_x, low_x - 1, -1):
            for row in range(high_y, low_y - 1, -1):
                index_pairs.append(((pixel_coord[0] + row, pixel_coord[1] + col),
                                    (row + high_y, col + high_x)))

    return index_pairs


def convolution(f, I):
    filter_x_dim = int(f.shape[1])
    filter_y_dim = int(f.shape[0])
    out_image = np.zeros(I.shape, dtype=np.float32)

    assert filter_x_dim == filter_y_dim, \
        "The convolution filter must " \
        "be an odd number and equal in " \
        "length on both sides."
    assert filter_x_dim % 2 == 1 and filter_y_dim % 2 == 1, \
        "The filter must be an odd " \
        "number of pixels."

    # STEP 1: add padding to the image
    number_pad_layers = int(np.floor(filter_x_dim / 2))
    padded_image = zero_padding(number_pad_layers, I)

    # STEP 2: Convolve the image
    for row in range(out_image.shape[0]):
        for col in range(out_image.shape[1]):
            op_order = get_operation_order((row, col),
                                           (filter_x_dim, filter_y_dim), False)
            for op in op_order:
                out_image[row][col] += \
                    padded_image[op[0][0] + number_pad_layers][op[0][1] + number_pad_layers] \
                    * f[op[1][0]][op[1][1]]

    return out_image


def calculate_gradient_magnitude(image_derivative_x, image_derivative_y):
    assert image_derivative_x.shape == image_derivative_y.shape, \
        "Image shapes should match for gradient magnitude"

    im_d_x_2 = np.square(image_derivative_x)
    im_d_y_2 = np.square(image_derivative_y)
    im_out = np.sqrt(im_d_x_2 + im_d_y_2)

    return im_out


def get_gradient_magnitude(image: np.ndarray):
    filter_deriv_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    filter_deriv_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

    im_der_x = convolution(filter_deriv_x, image)
    im_der_y = convolution(filter_deriv_y, image)

    return calculate_gradient_magnitude(im_der_x, im_der_y)


def create_gaussian_filter_2d(sigma: float):
    # How to choose size n
    n = 2 * math.floor(sigma * 3) + 1
    sigma2 = sigma * sigma

    gaussian_filter = np.zeros((n, n))

    x = np.linspace(-n / 2.0, n / 2.0, n)
    y = np.linspace(-n / 2.0, n / 2.0, n)

    for i in range(0, len(x)):
        for j in range(0, len(y)):
            gaussian_filter[i, j] = \
                (1 / (2 * math.pi * sigma2)
                 * math.exp(-(x[i] * x[i] + y[i] * y[i]) / (2 * sigma2)))

    return gaussian_filter / np.sum(gaussian_filter.flatten())


def threshold_image(image: np.ndarray, threshold: float,
                    background_val: int, object_val: int):
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row][col] >= threshold:
                image[row][col] = background_val
            else:
                image[row][col] = object_val

    return image


def get_point_neighborhood(point: tuple):
    packed_tuples = get_operation_order(point, (3, 3), True)
    unpacked = []

    for n in packed_tuples:
        unpacked.append((n[0][0], n[0][1]))

    return unpacked



