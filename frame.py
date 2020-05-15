import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_functions import *
from line import *

# Define a class to process each frame of a video
class Frame():
    def __init__(self, mtx, dist, M, M_inv, img_size):
        
        self.mtx = mtx
        self.dist = dist
        
        self.M = M
        self.M_inv = M_inv
        
        self.left_line = Line(img_size[0])
        self.right_line = Line(img_size[0])
        
    def __call__(self, frame):
        
        ## 1 - Apply a distortion correction to raw images ##

        # Undistort the image
        undist = undistort_image(frame, self.mtx, self.dist)

        ## 2 - Use color transforms, gradients, etc., to create a thresholded binary image ##
        ksize = 3 # Sobel kernel size 

        # Apply each of the gradient thresholding functions
        gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(30, 150))
        grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(50, 200))
        mag_binary = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(30, 200))
        dir_binary = dir_threshold(undist, sobel_kernel=ksize, thresh=(0.7, 1.2))

        # Apply each of the color thresholding functions
        colors_binary = hls_select(undist, thresh=(170, 255))

        # Combine all of the thresholding binaries
        # Take the binary image and find the histogram peaks which are good candidates for the left and right lane lines

        combined = np.zeros_like(gradx)
        combined[(colors_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)) | ((gradx == 1) & (grady == 1)) ] = 1

        # 3 - Apply a perspective transform to rectify binary image ("birds-eye view") ##

        # Warp the image to a top-down view
        img_size = (undist.shape[1], undist.shape[0])
        binary_warped = cv2.warpPerspective(combined, self.M, img_size, flags=cv2.INTER_LINEAR)


        ## 4 - Detect lane pixels and fit to find the lane boundary ##

        # Create a sliding window and find out which activated pixels fall into the window
        leftx_base, rightx_base = self.find_base_lanes_position(binary_warped)
        
        if self.left_line.detected == True:
            self.left_line.search_around_poly(binary_warped)
        else:    
            self.left_line.first_fit_polynomial(binary_warped, leftx_base)
            
        if self.right_line.detected == True:
            self.right_line.search_around_poly(binary_warped)
        else:    
            self.right_line.first_fit_polynomial(binary_warped, rightx_base)

        ## 5 - Determine the curvature of the lane and vehicle position with respect to center ##

        left_curverad = self.left_line.measure_curvature_real(undist.shape[0], self.left_line.best_fit)
        right_curverad = self.right_line.measure_curvature_real(undist.shape[0], self.right_line.best_fit)

        radius_curvature = np.int((left_curverad + right_curverad)/2)

        rel_vehicle_position = measure_rel_vehicle_position(undist.shape, self.left_line.best_fit, self.right_line.best_fit)

        ## 6 - Warp the detected lane boundaries back onto the original image ##

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.allx, self.left_line.ally]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.allx, self.right_line.ally])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.M_inv, img_size) 


        ## 7 - Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position ##
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        
        return result
    
    def find_base_lanes_position(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint


        return leftx_base, rightx_base