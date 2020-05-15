import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_functions import *

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, img_shape):
        
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        
        self.n_iterations = 10
        
        self.count = 0
        
        self.line_y = np.linspace(0, img_shape-1, img_shape )
        
        # was the line detected in the last iteration?
        self.detected = False  
        
        # x values of the last n fits of the line
        self.recent_xfitted = []
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        self.allx = None  
        
        #y values for detected line pixels
        self.ally = None  
        
        
    def find_lane_pixels(self, binary_warped, base):
       
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        x_current = base

        # Create empty lists to receive the lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_x_low = x_current - self.margin 
            win_x_high = x_current + self.margin 

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_inds = ((nonzerox>=win_x_low) & (nonzerox<win_x_high) &
            (nonzeroy>=win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_inds) > self.minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds] 

  
    def update_poly(self):
        # Fit a second order polynomial to each using `np.polyfit'
        poly_fit = np.polyfit(self.allx, self.ally, 2 )
        
        # Get the difference in fit coefficients between last and new fits
        if self.current_fit is not None:
            self.diffs = poly_fit - self.current_fit 
            
        self.current_fit = poly_fit
        
        # Generate x and y values for plotting  
        try:
            x = self.current_fit[0]*self.line_y**2 + poly_fit[1]*self.line_y + poly_fit[2]
            if len(self.recent_xfitted) == self.n_iteration:
                self.recent_xfitted.pop(0)
            self.recent_xfitted.append(x)
        except TypeError:
            # Avoids an error
            print('The function failed to fit a line!')
            self.detected = False

        # Calculate the average x values of the fitted line over the last 'self.n_iteration' iterations
        self.bestx = np.mean(self.recent_xfitted)
        self.best_fit = np.polyfit(self.bestx, self.line_y, 2 )
    
    def first_fit_polynomial(self, binary_warped, basex):
        # Find the lane pixels first
        self.find_lane_pixels(binary_warped, basex)

        # Update the polynomial
        self.update_poly()
       
        self.detected = True
    
    def search_around_poly(self, binary_warped):

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        x_current = (self.current_fit[0]*nonzeroy**2) + (self.current_fit[1]*nonzeroy) + self.current_fit[2]

        win_x_low = x_current - self.margin 
        win_x_high = x_current + self.margin 


        lane_inds = ((nonzerox>=win_x_low) & (nonzerox<win_x_high)).nonzero()[0]

        # Again, extract left and right line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds] 

        # update the polynomial
        self.update_poly()
        
        self.detected = True


    def measure_curvature_real(self, y_eval, ploy_fit):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        a = ploy_fit[0]*(xm_per_pix/ym_per_pix**2)
        b = ploy_fit[1]*(xm_per_pix/ym_per_pix)


        # Implement the calculation of R_curve (radius of curvature)
        self.radius_of_curvature = ((1 + (2*a*y_eval + b)**2)**(3/2))/(np.abs(2*a)) 

        return self.radius_of_curvature
