import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import helper
import threshold
import tracking
import affinity
import spectral_clustering

def main():

    #Step 1: Read in image
    #frame_0 = helper.import_im('../src/images/FBMS_marple13/marple13_20.jpg')
    #frame_1 = helper.import_im('../src/images/FBMS_marple13/marple13_21.jpg')

    #Step 2: Threshold
    #threshold_frame_0 = threshold.threshold_eigenvalues(frame_0, 50000)
    #threshold_frame_1 = threshold.threshold_eigenvalues(frame_1, 50000)

    # frame_0 = helper.import_im('/Users/andrewweng/developer/optical-flow-segmentation-analysis/src/images/Marple13_eig/eig_marple13_20.jpg')
    # frame_1 = helper.import_im('/Users/andrewweng/developer/optical-flow-segmentation-analysis/src/images/Marple13_eig/eig_marple13_21.jpg')
    frame_0 = helper.import_im('C:/Users/russe/Dropbox/Wustenberg/!_Tandon/6643_Vision/Repos/rw2873_CV_Project_F/optical-flow-segmentation-analysis/src/images/Marple13_eig/eig_marple13_22.jpg')
    frame_1 = helper.import_im('C:/Users/russe/Dropbox/Wustenberg/!_Tandon/6643_Vision/Repos/rw2873_CV_Project_F/optical-flow-segmentation-analysis/src/images/Marple13_eig/eig_marple13_23.jpg')
    # frame_4 = helper.import_im('/Users/andrewweng/developer/optical-flow-segmentation-analysis/src/images/Marple13_eig/eig_marple13_24.jpg')

    # helper.display_im(frame_0)

    frames = [frame_0, frame_1]

    trajectories = tracking.create_trajectories(frames)
    
    print(len(trajectories))

    A = affinity.calculate_A(trajectories, gamma = 0.1)
    # print(A[190:196,190:196])
    np.savetxt("A_out.csv", A, delimiter=",")

    clustering = spectral_clustering.spectral_clustering(df=A, n_neighbors=3, n_clusters=3)
    print(clustering)

    for i in range(len(clustering)):
        trajectories[i].label = clustering[i]

    
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(frame_0, cmap='gray')

    for i in range(len(trajectories)):
        point = trajectories[i].history[0]
        label = trajectories[i].label
        col=point[1]
        row=point[0]
        if label == 0:
            plt.scatter(col,row,c='b')
        if label == 1:
            plt.scatter(col,row,c='y')
        if label == 2:
            plt.scatter(col,row,c='r')
        if label == 3:
            plt.scatter(col,row,c='g')
    
    plt.show()


if __name__ == "__main__":
    main()