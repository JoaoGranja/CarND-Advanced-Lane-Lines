import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper_functions import *

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, image_shape):
        
        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = 9
        # Width of the windows +/- margin
        self.margin = 50 #100
        # Minimum number of pixels found to recenter window
        self.minpix = 50
        # Iteration number to average the polynomial coefficients
        self.n_iteration = 10
        
        # Image size 
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        
        # y values of the line, spaced by 1 pixel
        self.line_y = np.linspace(0, self.image_height-1, self.image_height )
        
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
        
        self.count = 0
        
        
    def find_lane_pixels(self, binary_warped, base, out_img = None):
        """
        Find the x and y pixels for the lane line.
        Sliding windows will be used around starting points (base) passed as argument
        """
            
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
            
        counter_empty_win = 0
        
        self.line_y = np.linspace(0, self.image_height-1, self.image_height )
        
        # Step through the windows one by one
        for window in range(self.nwindows):

            # Identify window boundaries in x and y 
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - self.margin 
            win_x_high = x_current + self.margin 

            ##  For Visualization - Draw the windows on the visualization image ##
            if out_img is not None:
                cv2.rectangle(out_img,(win_x_low,win_y_low),
                (win_x_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window 
            good_inds = ((nonzerox>=win_x_low) & (nonzerox<win_x_high) &
            (nonzeroy>=win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)
            
            # If > minpix pixels, recenter next window on their mean position
            if len(good_inds) > self.minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))
                counter_empty_win = 0
            else:
                counter_empty_win +=1
                # if 4 sequence windows have few pixels, it is better to stop searching more
                if counter_empty_win == 4:
                    self.allx = []
                    self.ally = []
                    print("failed")
                    break
                
            if ((x_current - self.margin ) <= 0) or ((x_current + self.margin)>= self.image_width):
                #self.line_y = np.linspace(win_y_high, self.image_height-1, (self.image_height-win_y_high) )
                print("test")
                break

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds] 

  
    def update_poly(self, order = 2, out_img = None):
        """
        Append the polynomial x points to a list "recent_xfitted" and calculate the average x values of the fitted line over the last 'self.n_iteration' iterations. 
        """
        
        self.count += 1
        
        # Fit a polynomial to each using `np.polyfit'
        poly_fit = np.polyfit(self.ally, self.allx, order )
        
        # Get the difference in fit coefficients between last and new fits
        if self.current_fit is not None:
            self.diffs = poly_fit - self.current_fit 
            #if np.max(np.abs(self.diffs/poly_fit)) > 10:
            #    print("Error poly fit", self.diffs/poly_fit )
            
        self.current_fit = poly_fit
        
        # Generate x and y values for plotting  
        try:
            x = 0
            for i in range(order+1):
                x += self.current_fit[i]*self.line_y**(order-i)
            if len(self.recent_xfitted) == self.n_iteration:
                self.recent_xfitted.pop(0)
            
            
            if (len(self.recent_xfitted) > 1):
                if (len(x) < len(self.recent_xfitted[0])):
                    print("teste2", len(x),  len(self.recent_xfitted[0]))
                    x = np.concatenate([np.array([x[0]]*(len(self.recent_xfitted[0]) - len(x))), x])
                    print(len(x), len(self.recent_xfitted[0]))
                                   
            self.recent_xfitted.append(x)
        except TypeError:
            # Avoids an error
            print('The function failed to fit a line!')
            self.detected = False

        # Calculate the average x values of the fitted line over the last 'self.n_iteration' iterations
        self.bestx = np.mean(self.recent_xfitted, axis = 0)

        self.best_fit = np.polyfit(self.line_y, self.bestx[-len(self.line_y):], order )
        
        ##  For Visualization ##
        if out_img is not None:
            # Colors in the left and right lane regions
            out_img[self.ally, self.allx] = [255, 0, 0]

            # Plots the left and right polynomials on the lane lines
            plt.plot(self.bestx[-len(self.line_y):], self.line_y, color='yellow')
        
    
    def first_fit_polynomial(self, binary_warped, basex, order = 2, out_img = None):
        """
        Fit a polynomial with order "order" for the lane line based on the x,y pixels which fall on sliding windows. 
        """
        
        # Find the lane pixels first
        self.find_lane_pixels(binary_warped, basex, out_img)

        # Update the polynomial
        if len(self.allx) > 0 and len(self.ally)>0:
            # update the polynomial
            self.update_poly(order, out_img)
            self.detected = True
        else:
            self.detected = False
    
    def search_around_poly(self, binary_warped, order, out_img = None):
        """
        Fit a polynomial with order "order" for the lane line based on the x,y pixels which are around a lane line detected on previous frame
        """
        
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the area of search based on activated x-values 
        # within the +/- margin of our polynomial function 
        x_current = 0
        for i in range(order+1):
            x_current += self.best_fit[i]*nonzeroy**(order-i)

        win_x_low = x_current - self.margin 
        win_x_high = x_current + self.margin 

        lane_inds = ((nonzerox>=win_x_low) & (nonzerox<win_x_high)).nonzero()[0]

        # Again, extract left and right line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds] 
        
        if len(self.allx) > 0 and len(self.ally)>0:
            # update the polynomial
            self.update_poly(order, out_img)

            self.detected = True
        else:
            print("X,Y are empty")
            self.detected = False
            
        ##  For Visualization ##
        if out_img is not None:
            # Create an image to draw on and an image to show the selection window
            window_img = np.zeros_like(out_img)

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            line_window1 = np.array([np.transpose(np.vstack([self.bestx[-len(self.line_y):]-self.margin, self.line_y]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.bestx[-len(self.line_y):]+self.margin, 
                                      self.line_y])))])
            line_pts = np.hstack((line_window1, line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
