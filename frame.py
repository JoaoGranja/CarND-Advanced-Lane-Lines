import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_functions import *
from line import *
#from scipy.signal import find_peaks

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meters per pixel in x dimension
horizontal_distance = 600 #pixels

# Define a class to process each frame of a video
class Frame():
    def __init__(self, mtx, dist, M, M_inv, img_size):
        
        self.mtx = mtx
        self.dist = dist
        
        self.M = M
        self.M_inv = M_inv
        
        self.img_size = img_size
        
        # Two Line instances for each left and right lane line
        self.left_line = Line(img_size)
        self.right_line = Line(img_size)
        
        #radius of curvature of the line in meter unit
        self.radius_of_curvature = None
        
        #distance in meters of vehicle center from the lane line center. Positive value means is on left
        self.line_base_pos = 0 
        
        # Flag to indicate if first_fit_polynomial shall be called for each line.
        self.search_starting_points = True
        
    def __call__(self, frame):
        
        ## 1 - Apply a distortion correction to the image ##
        undist = undistort_image(frame, self.mtx, self.dist)

        ## 2 - Use color transforms, gradients, etc., to create a thresholded binary image ##
        ksize = 3 # Sobel kernel size 

        # Apply each of the gradient thresholding functions
        gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(30, 200))

        # Apply each of the color thresholding functions
        colors_binary = hls_select(undist, s_thresh=(200, 255), v_thresh=(230, 255))

        # Combine all of the thresholding binaries
        binary_image = np.zeros_like(gradx)
        binary_image[(colors_binary == 1) | (gradx == 1) ] = 1
        
        # 3 - Apply a perspective transform to rectify binary image ("birds-eye view") ##

        # Warp the image to a top-down view
        binary_warped = cv2.warpPerspective(binary_image, self.M, (self.img_size[1],self.img_size[0]) , flags=cv2.INTER_LINEAR)
        
        ## 4 - Detect lane pixels and fit to find the lane boundary ##

        # Create a sliding window and find out which activated pixels fall into the window
        if self.search_starting_points:
            leftx_base, rightx_base = self.find_base_lanes_position(binary_warped)
            
            self.left_line.first_fit_polynomial(binary_warped, leftx_base)
            self.right_line.first_fit_polynomial(binary_warped, rightx_base)
            
            self.search_starting_points = False
        else:
            self.left_line.search_around_poly(binary_warped)
            self.right_line.search_around_poly(binary_warped)

        ## 5 - Determine the curvature of the lane and vehicle position with respect to center ##

        self.left_line.measure_curvature_real()
        self.right_line.measure_curvature_real()
        #self.radius_of_curvature = np.int((self.left_line.radius_of_curvature + self.right_line.radius_of_curvature)/2)
        
        self.radius_of_curvature = measure_curvature_real(self.img_size[0], self.left_line.best_fit, self.right_line.best_fit)

        self.line_base_pos = measure_rel_vehicle_position(self.img_size, self.left_line.best_fit, self.right_line.best_fit)
             
        ## Sanity Check ##
        # A - Checking that they have similar curvature       
        if not 0.5 <(self.left_line.radius_of_curvature / self.right_line.radius_of_curvature)< 2:
            print("error 1", self.right_line.radius_of_curvature,  self.left_line.radius_of_curvature)
        
        # B - Checking that they are separated by approximately the right distance horizontally
        distance = (self.right_line.bestx[self.img_size[0]-1] - self.left_line.bestx[self.img_size[0]-1])
        if not (0.8*horizontal_distance<distance<1.2*horizontal_distance):
            print("error 2", distance, self.right_line.bestx[self.img_size[0]-1], self.left_line.bestx[self.img_size[0]-1])
            self.search_starting_points = True
            
        distance = (self.right_line.bestx[0] - self.left_line.bestx[0])
        if not (0.6*horizontal_distance<distance<1.4*horizontal_distance):
            print("error 4", distance, self.right_line.bestx[0], self.left_line.bestx[0])
            self.search_starting_points = True
        
        # C - Checking that they are roughly parallel

        ## 6 - Warp the detected lane boundaries back onto the original image ##

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, self.left_line.line_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, self.right_line.line_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.M_inv, (self.img_size[1],self.img_size[0])) 


        ## 7 - Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position ##
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        
        # Write some Text
        cv2.putText(result,"Radius of the Curvature = {0}m".format(self.radius_of_curvature), 
            (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        if self.line_base_pos < 0:
            cv2.putText(result,"Vehicle is {0}m left of the center".format(-self.line_base_pos), 
            (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        else:    
            cv2.putText(result,"Vehicle is {0}m right of the center".format(self.line_base_pos), 
            (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
        return result
    
    def find_base_lanes_position(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Find the peaks of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)  
        
        # The idea is to create a list of left and right peaks. 
        # Then select the peaks each are roughly spaced by "horizontal_distance"
        # and have the highest peak values
        leftx_list = []
        rightx_list = []        
        leftx_list.append(np.argmax(histogram[:midpoint]))
        rightx_list.append(np.argmax(histogram[midpoint:]) + midpoint)
        
        for i in range(2):
            # Left peaks
            window_low = leftx_list[i] - 25
            if window_low < 0:
                window_low = 0
        
            window_high = leftx_list[i] + 25
            if window_high > midpoint:
                window_high = midpoint
            histogram[window_low:window_high] = 0
            leftx_list.append(np.argmax(histogram[:midpoint]))
            
            # Right peaks
            window_low = rightx_list[i] - 25
            if window_low < midpoint:
                window_low = midpoint
        
            window_high = rightx_list[i] + 25
            if window_high > histogram.shape[0]:
                window_high = histogram.shape[0]
            histogram[window_low:window_high] = 0
            rightx_list.append(np.argmax(histogram[midpoint:]) + midpoint)
            
            
        # Find peaks where distance is similar to horizontal_distance
        leftx_base, rightx_base = leftx_list[0], rightx_list[0]
        maximum_peaks = histogram[leftx_base] + histogram[rightx_base]
        first_time = True
        
        for i in range(2):
            for j in range(2):
                if 0.8*horizontal_distance<rightx_list[j] - leftx_list[i]<1.2*horizontal_distance:
                    #print("Find peaks", rightx_list[j], leftx_list[i])
                    if first_time | (histogram[leftx_list[i]] + histogram[rightx_list[j]]) > maximum_peaks:
                        leftx_base, rightx_base = leftx_list[i], rightx_list[j]
                        maximum_peaks = histogram[leftx_base] + histogram[rightx_base]
                        first_time = False
        
        #print("returned peaks", leftx_base, rightx_base, maximum_peaks)
        return leftx_base, rightx_base
        
    
    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        ignore_mask_color = 1

        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image