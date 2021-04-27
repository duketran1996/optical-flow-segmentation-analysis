#!/usr/bin/python3
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import time
import os

MAX_GREYSCALE = 255

#Create Gaussian filter
def create_gaussian_filter(sigma):
    # How to choose size n
    n = 2*math.floor(sigma*3)+1
    # Precompute sigma*sigma
    sigma2 = sigma*sigma
    
    # Create a coordinate sampling from -n/2 to n/2 so that (0,0) will be at the center of the filter
    x = np.linspace(-n/2.0, n/2.0, n)
    y = np.linspace(-n/2.0, n/2.0, n)
    
    # Blank array for the Gaussian filter
    gaussian_filter = np.zeros((n,n))

    # Loop over all elements of the filter
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            # Use the x and y coordinate sampling as the inputs to the 2D Gaussian function
            gaussian_filter[i,j] = (1/(2*math.pi*sigma2))*math.exp(-(x[i]*x[i]+y[j]*y[j])/(2*sigma2))
        
    # Normalize so the filter sums to 1
    return gaussian_filter/np.sum(gaussian_filter.flatten())

def convert_im_grayscale(im):
    return ((im - np.min(im)) * (1/(np.max(im) - np.min(im)) * MAX_GREYSCALE)).astype('uint8')

def blur_img(im_in, sigma_blur):
    g = create_gaussian_filter(sigma_blur)
    blur_im = cv.filter2D(im_in, -1 ,g)
    return blur_im

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

def vid_denoise(frames):
    frames_denoise = []
    for f in frames:
        frames_denoise.append(blur_img(f, 1))

    return frames_denoise

def vid_subsample(frames):
    frames_subsample = []
    for f in frames:
        frames_subsample.append(f[::2,::2])

    return frames_subsample

def vid_edge(frames_denoise, grayscale):
    frames_edge = []
    for f in frames_denoise:
        gradx = cv.Sobel(f,cv.CV_64F,1,0,ksize=3)
        grady = cv.Sobel(f,cv.CV_64F,0,1,ksize=3)
        gradient_mag_im = cv.magnitude(gradx,grady)
        if grayscale:
            gradient_mag_im =  convert_im_grayscale(gradient_mag_im)

        frames_edge.append(gradient_mag_im)

    return [gradx, grady, frames_edge]

def temporal_derivative(frame1, frame2):
    return frame2 - frame1

def lucas_kanade(Ix, Iy, It, kernel_size):
    h, w = Ix.shape
    kernal_length = kernel_size // 2
    flow = {}
    for row in range(kernal_length, h-kernal_length):
        for col in range(kernal_length, w-kernal_length):
            sub_Ix = Ix[row-kernal_length:row+kernal_length+1, col-kernal_length:col+kernal_length+1].reshape(-1, 1)
            sub_Iy = Iy[row-kernal_length:row+kernal_length+1, col-kernal_length:col+kernal_length+1].reshape(-1, 1)
            sub_It = It[row-kernal_length:row+kernal_length+1, col-kernal_length:col+kernal_length+1].reshape(-1, 1) * (-1)
            m1 = np.hstack([sub_Ix, sub_Iy])
            u, v = np.linalg.lstsq(m1, sub_It, rcond=None)[0]
            flow[tuple((row,col))] = [u[0],v[0]]

    return flow

def convert_im_grayscale(im):
    return ((im - np.min(im)) * (1/(np.max(im) - np.min(im)) * MAX_GREYSCALE)).astype('uint8')
    
def im_magnitude(size, flow):
    im_mag = np.zeros(size)

    for key, val in flow.items():
        vectors = np.array(list(val))
        points = np.array(list(key))

        mag = np.sqrt(np.sum(vectors.dot(vectors.T)))
    
        im_mag[points[0], points[1]] = mag*100

    return im_mag

def compute_for_frames(frames_subsample, frames_edge):

    im_mag_frames = []
    for i in range(0, len(frames_subsample)-1):
        It = temporal_derivative(frames_subsample[i], frames_subsample[i+1])

        Ix = frames_edge[i]

        Iy = frames_edge[i+1]

        flow = lucas_kanade(Ix, Iy, It, 5)

        #vectors = np.array(list(flow.values()))
        #points = np.array([list(point) for point in flow.keys()])

        #vector_normalize = vectors/np.linalg.norm(vectors)

        im_mag = convert_im_grayscale(im_magnitude(frames_subsample[0].shape, flow))
        im_mag_frames.append(im_mag)
        
        #step = 50
        #plt.quiver(points[::step,0], points[::step,1], vector_normalize[::step,0], vector_normalize[::step,1], scale=1, color='b')
        #plt.title('Optical Flow')
        #plt.savefig('./lucas_result/' + 'result' + str(i))
        #plt.show()

    vid_play(im_mag_frames, True, './lucas_mag_result/')


def main():

    frames = vid_in('../src/videos/FBMS_marple13.mov', 1, 75)

    frames_denoise = vid_denoise(frames)

    frames_subsample = vid_subsample(frames_denoise)

    frames_edge = vid_edge(frames_subsample, True)

    compute_for_frames(frames_subsample, frames_edge[2])

    #It = temporal_derivative(frames[0], frames[1])

    #Ix = frames_edge[0]

    #Iy = frames_edge[1]

    #vid_play(frames_edge[2], False)

    #flow = lucas_kanade(Ix, Iy, It, 5)

    #vector = np.array(list(flow.values()))
    #points = np.array([list(point) for point in flow.keys()])

    #mag, ang = cv.cartToPolar(vector[...,0], vector[...,1])

    #vector_normalize = vector/np.linalg.norm(vector)



    # hsv = np.zeros_like(frames[0])
    # hsv[...,1] = 255
    #mag, ang = cv.cartToPolar(vector[...,0], vector[...,1])
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    # bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    # cv.imshow('frame2',bgr)

    #mag, ang = cv.cartToPolar(vector[:,0], vector[:,1])

    #ang =  ang*180/np.pi/2
    #mag = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)

    #step = 50
    #plt.quiver(points[::step,0], points[::step,1], vector[::step,0], vector[::step,1], scale=250, color='b')
    
    #plt.quiver(points[::step,0], points[::step,1], vector_normalize[::step,0], vector_normalize[::step,1], scale=1, color='b')
    #plt.title('Optical Flow')

    #plt.show()

    #vid_play(frames_edge)

    # # Denoise
    # denoise_im = blur_img(im_in, 1)

    # # Use open cv for filtering process and calculater gradient magnitude
    # gradx = cv.Sobel(denoise_im,cv.CV_64F,1,0,ksize=3)
    # grady = cv.Sobel(denoise_im,cv.CV_64F,0,1,ksize=3)
    # gradient_mag_im = cv.magnitude(gradx,grady)
    # gradient_mag_im = convert_im_grayscale(gradient_mag_im)


if __name__ == "__main__":
    main()
