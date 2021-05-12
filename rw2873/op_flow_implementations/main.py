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
    tracks = []
    frames = []

    flow_fore = np.zeros(frames[0].shape, dtype=np.float32)
    flow_back = np.zeros(frames[0].shape, dtype=np.float32)

    for frame in range(0,len(frames)):
        # Calculate the forward and backwards flow between the current and next frame
        flow_fore = cv.calcOpticalFlowFarneback(frames[frame], frames[frame + 1], flow_fore,
                                                0.5, 5, 5, 5, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow_back = cv.calcOpticalFlowFarneback(frames[frame + 1], frames[frame], flow_back,
                                                0.5, 5, 5, 5, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)

        # Check if new objects are in scene in the FORWARD FLOW
        for row in flow_fore:
            for col in flow_fore:
                # check each pixel for flow response

                # if there is a response > than a certain threshold
                for track in tracks: # check track.curr_pos
                # if no track is currently tracking the point, create a new one

        for curr_track in tracks:
            fwd_flow = flow_fore[curr_track.curr_position[0]][curr_track.curr_position[1]]

            # New position is (x + u, y + v)
            new_pos = (curr_track.curr_position[0] + fwd_flow[0], curr_track.curr_position[1] + fwd_flow[1])


            bck_flow_u = # BILINEAR INTERPOLATION ON NEW_POS ON U
            bck_flow_v = # BILINEAR INTERPOLATION ON NEW_POST ON V

            # Check to make sure they match (using the W equation)

            # Update & append position if they do, kill if not

    # STEP 3) Construct affinity matrix

    # STEP 4) Populate affinity values

    # Step 5) Spectral Clustering






    return 0


if __name__ == '__main__':
    main()
