#########################################
#   Optical Flow Implementations        #
#   CS-GY 6643, Computer Vision         #
#   Professor James Fishbaugh           #
#   Author:        Russell Wustenberg   #
#   Collaborators: Duc Tran             #
#                  Andrew Weng          #
#                  Ye Xu                #
#########################################

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot
import math

import image_functions

image_1 = image_functions.open_image('sources/images/marple13_20.jpg')
image_2 = image_functions.open_image('sources/images/marple13_21.jpg')

image_2 = image_functions.output_intensity_mapping(image_2)

im_2_gauss_blur = cv.bilateralFilter(image_2, 9, 200, 50)

image_functions.output_image(im_2_gauss_blur, 'test_im2_bi_009_200_050.png')




