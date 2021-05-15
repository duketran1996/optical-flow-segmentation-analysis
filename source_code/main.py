import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import helper
import threshold

def main():

    #Step 1: Read in image
    #frame_0 = helper.import_im('../src/images/FBMS_marple13/marple13_20.jpg')
    #frame_1 = helper.import_im('../src/images/FBMS_marple13/marple13_21.jpg')

    #Step 2: Threshold
    #threshold_frame_0 = threshold.threshold_eigenvalues(frame_0, 50000)
    #threshold_frame_1 = threshold.threshold_eigenvalues(frame_1, 50000)

    frame_0 = helper.import_im('../src/images_threshold/marple13_eig/eig_marple13_20.jpg')
    frame_1 = helper.import_im('../src/images_threshold/marple13_eig/eig_marple13_21.jpg')

    helper.display_im(frame_0)

if __name__ == "__main__":
    main()