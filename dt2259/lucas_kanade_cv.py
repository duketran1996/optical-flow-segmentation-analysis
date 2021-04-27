import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt

MAX_GREYSCALE = 255

def convert_im_grayscale(im):
    return ((im - np.min(im)) * (1/(np.max(im) - np.min(im)) * MAX_GREYSCALE)).astype('uint8')

def vid_in(filename, from_frame, to_frame):

    frames = []
    cap = cv.VideoCapture(filename)

    count = 1
    switch = False
    while True:
        ret, frame = cap.read()
        if count == from_frame or switch:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frames.append(gray)
            switch = True

        if count == to_frame:
            return frames

        count += 1

def vid_play(frames, write, path):
    count = 0
    for f in frames:
        cv.imshow('frame', f)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)
        if write:
            cv.imwrite(path + "image_" + str(count) + '.png', f)
        count += 1

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def im_magnitude(size, flow):
    im_mag = np.zeros(size)

    for key, val in flow.items():
        vectors = np.array(list(val))
        points = np.array(list(key))

        mag = np.sqrt(np.sum(vectors.dot(vectors.T)))
    
        im_mag[int(points[1]), int(points[0])] = mag*100

    return im_mag

def lucas_kanade_cv(frames):
    
    frames_mag = []

    for i in range(0, len(frames)-1):

        # Get the coordinates of frames
        grid_y, grid_x = np.mgrid[0:frames[i].shape[0]:1, 0:frames[i].shape[1]:1]
        # Stack grid_y and grid_x to 1 array
        p0 = np.stack((grid_x.flatten(),grid_y.flatten()),axis=1).astype(np.float32)

        # Calculate optical flow
        # Return p1 which is the point original point from frames[i] will move to.
        p1, st, err = cv.calcOpticalFlowPyrLK(frames[i], frames[i+1], p0, None, **lk_params)

        # Reshape to 2D array with size of frame.
        flow = np.reshape(p1 - p0, (frames[i].shape[0], frames[i].shape[1], 2))
        points = np.reshape(p0, (frames[i].shape[0], frames[i].shape[1], 2))
        
        # Build dictionary with key as point and value as flow uv vector
        flow_dict = {} 
        item = list(zip(flow.reshape(-1, 2), points.reshape(-1, 2)))
        for i in item:
            flow_dict[tuple(i[1])] = i[0]

        # Get the magnitude and convert to grayscale
        im_mag = convert_im_grayscale(im_magnitude(frames[0].shape, flow_dict))
        frames_mag.append(im_mag)


        #step = 50
        #plt.quiver(points[::step,0], points[::step,1], flow[::step,0], flow[::step,1], scale=100, color='b')
        #plt.title('Optical Flow')
        #plt.savefig('./lucas_result/' + 'result' + str(i))
        #plt.show()

    return frames_mag



def main():

    frames = vid_in('../src/videos/FBMS_marple13.mov', 1, 75)

    frames_mag = lucas_kanade_cv(frames)

    # Play mag frames
    vid_play(frames_mag, True, './lucas_mag_cv_result/')


if __name__ == "__main__":
    main()