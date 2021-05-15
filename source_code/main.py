import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import helper
import threshold
import tracking
import affinity

def main():

    #Step 1: Read in image
    #frame_0 = helper.import_im('../src/images/FBMS_marple13/marple13_20.jpg')
    #frame_1 = helper.import_im('../src/images/FBMS_marple13/marple13_21.jpg')

    #Step 2: Threshold
    #threshold_frame_0 = threshold.threshold_eigenvalues(frame_0, 50000)
    #threshold_frame_1 = threshold.threshold_eigenvalues(frame_1, 50000)

    frame_0 = helper.import_im('/Users/andrewweng/developer/optical-flow-segmentation-analysis/src/images/Marple13_eig/eig_marple13_20.jpg')
    frame_1 = helper.import_im('/Users/andrewweng/developer/optical-flow-segmentation-analysis/src/images/Marple13_eig/eig_marple13_21.jpg')

    # helper.display_im(frame_0)

    frames = [frame_0, frame_1]

    trajectories = tracking.create_trajectories(frames)
    
    print(len(trajectories))

    A = affinity.calculate_A(trajectories, gamma = 0.1)
    print(A[:100,:100])



if __name__ == "__main__":
    main()