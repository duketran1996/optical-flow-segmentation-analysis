#!/usr/bin/python3
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import time

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

def vid_in(filename):

    frames = []
    cap = cv.VideoCapture(filename)

    for _ in range(0,5):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(gray)
    
    return frames
        
def vid_play(frames):
    for f in frames:
        cv.imshow('frame', f)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)

def vid_denoise(frames):
    frames_denoise = []
    for f in frames:
        frames_denoise.append(blur_img(f, 1))

    return frames_denoise

def vid_edge(frames_denoise, grayscale):
    frames_edge = []
    for f in frames_denoise:
        gradx = cv.Sobel(f,cv.CV_64F,1,0,ksize=3)
        grady = cv.Sobel(f,cv.CV_64F,0,1,ksize=3)
        gradient_mag_im = cv.magnitude(gradx,grady)
        if grayscale:
            gradient_mag_im =  convert_im_grayscale(gradient_mag_im)

        frames_edge.append(gradient_mag_im)

    return frames_edge


def main():

    frames = vid_in('walk.mp4')

    frames_denoise = vid_denoise(frames)

    frames_edge = vid_edge(frames_denoise, True)

    vid_play(frames_edge)

    # # Denoise
    # denoise_im = blur_img(im_in, 1)

    # # Use open cv for filtering process and calculater gradient magnitude
    # gradx = cv.Sobel(denoise_im,cv.CV_64F,1,0,ksize=3)
    # grady = cv.Sobel(denoise_im,cv.CV_64F,0,1,ksize=3)
    # gradient_mag_im = cv.magnitude(gradx,grady)
    # gradient_mag_im = convert_im_grayscale(gradient_mag_im)


if __name__ == "__main__":
    main()
