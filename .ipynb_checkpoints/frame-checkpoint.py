import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_functions import *
from line import *

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meters per pixel in x dimension
horizontal_distance = 600 #pixels

out_img = None

# Define a class to process each frame of a video
class Frame():
    def __init__(self, mtx, dist, M, M_inv, img_size, order, debug=False):
        
        self.mtx = mtx
        self.dist = dist
        
        self.M = M
        self.M_inv = M_inv
        
        self.img_size = img_size
        
        # Order of the polynomial of the lane
        self.order_poly = order
        
        #Save the frame image after self.count_max iterations 
        self.debug = debug
        self.count = 0 
        self.count_max = 10
        
        # Two Line instances for each left and right lane line
        self.left_line = Line(img_size, debug)
        self.right_line = Line(img_size, debug)
        
        #radius of curvature of the line in meter unit
        self.radius_of_curvature = None
        
        #distance in meters of vehicle center from the lane line center. Positive value means is on left
        self.line_base_pos = 0 
        
        # Flag to indicate if first_fit_polynomial shall be called for each line.
        self.search_starting_points = True
        
        # Clahe object to perform Adaptive Histogram Equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def __call__(self, frame):
        self.count += 1 
        
        ## 1 - Apply a distortion correction to the image ##
        undist = undistort_image(frame, self.mtx, self.dist)

        ## 2 - Use color transforms, gradients, etc., to create a thresholded binary image ##
        ksize = 3 # Sobel kernel size 

        # Apply each of the gradient thresholding functions
        mag_binary = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(30, 200))
        dir_binary = dir_threshold(undist, sobel_kernel=ksize, thresh=(0.7, 1.3))

        # Apply each of the color thresholding functions for HLS color space
        hls_colors_binary = hls_select(undist, s_thresh=(170, 250), l_thresh=(200, 255))

        # Apply each of the color thresholding functions for HSV color space
        hsv_colors_binary = hsv_select(undist, s_thresh=(130, 255), v_thresh=(240, 255), vs_thresh=(200, 255), clahe = self.clahe) 
         

        # Combine all of the thresholding binaries
        binary_image = np.zeros_like(mag_binary)
        binary_image[(hsv_colors_binary == 1) | (hls_colors_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)) ] = 1
        #binary_image[(hsv_colors_binary == 1) | (hls_colors_binary == 1)] = 1
        
        # 3 - Apply a perspective transform to rectify binary image ("birds-eye view") ##

        # Warp the image to a top-down view
        binary_warped = cv2.warpPerspective(binary_image, self.M, (self.img_size[1],self.img_size[0]) , flags=cv2.INTER_LINEAR)
        
        # Save the frame images
        if (self.debug) & ((self.count % self.count_max) == 0) :
            warped = cv2.warpPerspective(undist, self.M, (self.img_size[1],self.img_size[0]) , flags=cv2.INTER_LINEAR)
            save_warped_images("output_images/test_images/challenge", "binary" + str(self.count), binary_warped, warped)
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        else:
            out_img = None
        
        ## 4 - Detect lane pixels and fit to find the lane boundary ##

        # Create a sliding window and find out which activated pixels fall into the window
        if (self.search_starting_points == True) or ((self.count % 5) == 0): # | (self.left_line.detected == False) | (self.right_line.detected == False):
            leftx_base, rightx_base = self.find_base_lanes_position(binary_warped)
            
            self.left_line.first_fit_polynomial(binary_warped, leftx_base, self.order_poly, out_img)
            self.right_line.first_fit_polynomial(binary_warped, rightx_base, self.order_poly, out_img)
            
            self.search_starting_points = False
        else:
            self.left_line.search_around_poly(binary_warped, self.order_poly, out_img)
            self.right_line.search_around_poly(binary_warped, self.order_poly, out_img)
            
        # Save the frame images
        if (self.debug) & ((self.count % self.count_max) == 0) :
            save_lane_lines_image("output_images/test_images/challenge", "challenge" + str(self.count) , out_img)

        ## 5 - Determine the curvature of the lane and vehicle position with respect to center ##
        self.radius_of_curvature, self.left_line.radius_of_curvature, self.right_line.radius_of_curvature = measure_curvature_real(self.img_size[0], self.left_line.best_fit, self.right_line.best_fit)

        self.line_base_pos = measure_rel_vehicle_position(self.img_size, self.left_line.best_fit, self.right_line.best_fit)
             
        ## Sanity Check ##
        # A - Checking that they have similar curvature except when it seems it is a straight line
        if (self.radius_of_curvature < 2000) and (not 0.2 <(self.left_line.radius_of_curvature / self.right_line.radius_of_curvature) < 5):
            if self.debug:
                print("Fail sanity check - curvature", "left curvature:", self.right_line.radius_of_curvature,  "right curvature:", self.left_line.radius_of_curvature)
        
        # B - Checking that they are separated by approximately the right distance horizontally
        distance = (self.right_line.bestx[self.img_size[0]-1] - self.left_line.bestx[self.img_size[0]-1])
        if not (0.8*horizontal_distance<distance<1.2*horizontal_distance):
            if self.debug:
                print("Fail sanity check - distance at the bottom", distance, "right x point:", self.right_line.bestx[self.img_size[0]-1], "left x point:", self.left_line.bestx[self.img_size[0]-1])
                
            self.search_starting_points = True
        
        # C - Checking that they are roughly parallel
        min_distance = np.min(self.right_line.bestx - (self.left_line.bestx))
        if (min_distance < 0.8*horizontal_distance):
            if self.debug:
                print("Fail sanity check - roughly parallel (minimum distance)", min_distance)
            self.search_starting_points = True
                              
        max_distance = np.max(self.right_line.bestx - (self.left_line.bestx))
        if (max_distance > 1.25*horizontal_distance):
            if self.debug:
                print("Fail sanity check - roughly parallel (maximum distance)", max_distance)
            self.search_starting_points = True
                              
        mean_distance = np.mean(self.right_line.bestx - (self.left_line.bestx))
        if not (0.8*horizontal_distance<mean_distance<1.25*horizontal_distance):
            if self.debug:
                print("Fail sanity check - roughly parallel (mean distance)", mean_distance)
            self.search_starting_points = True
                        
        # D - Checking that vehicle is roughly close to the lane center
        if np.abs( self.line_base_pos) > 0.5:
            if self.debug:
                print("Fail sanity check - vehicle position from the lane center", self.line_base_pos)
            self.search_starting_points = True


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

        if self.line_base_pos > 0:
            cv2.putText(result,"Vehicle is {0}m left of the center".format(self.line_base_pos), 
            (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        else:    
            cv2.putText(result,"Vehicle is {0}m right of the center".format(-self.line_base_pos), 
            (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
        return result
    
    def find_base_lanes_position(self, binary_warped):
        """
        Find the x starting points for searching the left and right lane lines.
        """
        margin = 25 
        
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Find the peaks of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)  
        
        # The idea is to create a list of left and right peaks. 
        # Then select the peaks each are roughly spaced by "horizontal_distance"
        # and are closest to the image center
        leftx_list = []
        rightx_list = []        
        leftx_list.append(np.argmax(histogram[:midpoint]))
        min_left_peak = np.max(histogram[:midpoint])
        rightx_list.append(np.argmax(histogram[midpoint:]) + midpoint)
        min_right_peak = np.max(histogram[midpoint:])
        
        for i in range(2):
            # Left peaks
            window_low = leftx_list[len(leftx_list)-1] - margin
            if window_low < 0:
                window_low = 0
        
            window_high = leftx_list[len(leftx_list)-1] + margin
            if window_high > midpoint:
                window_high = midpoint
            histogram[window_low:window_high] = 0
            
            if np.max(histogram[:midpoint]) > 0.5 * min_left_peak:
                min_left_peak = np.min([min_left_peak, np.max(histogram[:midpoint])])
                leftx_list.append(np.argmax(histogram[:midpoint]))
            
            # Right peaks
            window_low = rightx_list[len(rightx_list)-1] - margin
            if window_low < midpoint:
                window_low = midpoint
        
            window_high = rightx_list[len(rightx_list)-1] + margin
            if window_high > histogram.shape[0]:
                window_high = histogram.shape[0]
            histogram[window_low:window_high] = 0
            
            if np.max(histogram[midpoint:]) > 0.5 * min_right_peak:
                min_right_peak = np.min([min_right_peak, np.max(histogram[midpoint:])])
                rightx_list.append(np.argmax(histogram[midpoint:]) + midpoint)
            
            
        # Find peaks where distance is similar to horizontal_distance
        leftx_base, rightx_base = leftx_list[0], rightx_list[0]
        min_distance = np.abs(binary_warped.shape[1]//2 - (rightx_list[0] + leftx_list[0])/2)
        first_time = True
        
        for i in range(len(leftx_list)):
            for j in range(len(rightx_list)):
                if 0.9*horizontal_distance<(rightx_list[j] - leftx_list[i])<1.1*horizontal_distance:    
                    if first_time | (np.abs(binary_warped.shape[1]/2 - (rightx_list[j] + leftx_list[i])/2) < min_distance ) :
                        leftx_base, rightx_base = leftx_list[i], rightx_list[j]
                        min_distance = np.abs(binary_warped.shape[1]//2 - (rightx_list[j] + leftx_list[i])/2)
                        first_time = False
        
        return leftx_base, rightx_base
