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

def main():
    frame_1 = image_functions.open_image('./sources/images/marple13_20.jpg')
    frame_2 = image_functions.open_image('./sources/images/marple13_21.jpg')

    test = image_functions.threshold_eigenvalues(frame_1, 0.75, 5)
    image_functions.display_image(test)


    return 0


if __name__ == '__main__':
    main()
