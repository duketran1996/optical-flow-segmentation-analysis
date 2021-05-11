#########################################
#   Optical Flow Implementations        #
#   CS-GY 6643, Computer Vision         #
#   Professor James Fishbaugh           #
#   Author:        Russell Wustenberg   #
#   Collaborators: Duc Tran             #
#                  Andrew Weng          #
#                  Ye Xu                #
#########################################
import numpy as np
import image_functions
import draw
import cv2 as cv
import track
import os

def main():
    path = './sources/images/FBMS_marple13'
    directory = os.fsencode(path)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.jpg'):
            img = image_functions.open_image(path + '/' + filename)
            im_eig = image_functions.threshold_eigenvalues(img, 0.2, 5)
            image_functions.output_image(im_eig, 'eig_' + filename)
        else:
            continue

    return 0


if __name__ == '__main__':
    main()
