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

def main():
    n = image_functions.get_neighborhood(5, 5, ksize=3)
    print(n)

if __name__ == '__main__':
    main()




