import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def undistort_image(image, mtx, dist):
    """Undistort it based on camera coefficients"""

    # Undistort the image an display it 
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return  undist


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where threshold are met
    grad_binary = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1]) ] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude of the gradient 
    mag_sobel = np.sqrt(sobelx**2, sobelx**2)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    # 5) Create a binary mask where the magnitude thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    # 6) Return this mask as your binary_output image
    mag_binary[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1]) ] = 1

    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(dir_sobel)
    # 6) Return this mask as your binary_output image
    dir_binary[(dir_sobel > thresh[0]) & (dir_sobel < thresh[1]) ] = 1
    
    return dir_binary

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def corners_unwarp(img, nx, ny):
    # Pass in your image into this function

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # 3) If corners found: 
    if ret == True:
            
            # a) draw corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
            src = np.float32([corners[0],corners[7],corners[16],corners[23]])
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            dst = np.float32([[80,80],[1200,80],[80,400],[1200,400]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # e) use cv2.warpPerspective() to warp your image to a top-down view
            img_size = (undist.shape[1], undist.shape[0])
            warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR) 
    return warped, M


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin # Update this
        win_xleft_high = leftx_current + margin # Update this
        win_xright_low = rightx_current - margin   # Update this
        win_xright_high = rightx_current + margin   # Update this
        
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high) &
        (nonzeroy>=win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox>=win_xright_low) & (nonzerox<win_xright_high) &
        (nonzeroy>=win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
        
        #print(good_left_inds)
      
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds])) 

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, save_image=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2 )
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    if save_image:
        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, left_fitx, right_fitx, ploty

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2 )
    right_fit = np.polyfit(righty, rightx, 2 )
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = (left_fit[0]*ploty**2) + (left_fit[1]*ploty) + left_fit[2]
    right_fitx = (right_fit[0]*ploty**2) + (right_fit[1]*ploty) + right_fit[2]
    
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    leftx_current = (left_fit[0]*nonzeroy**2) + (left_fit[1]*nonzeroy) + left_fit[2]
    rightx_current = (right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + right_fit[2]
    
    win_xleft_low = leftx_current - margin # Update this
    win_xleft_high = leftx_current + margin # Update this
    win_xright_low = rightx_current - margin   # Update this
    win_xright_high = rightx_current + margin   # Update this
    

    left_lane_inds = ((nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high)).nonzero()[0]
    right_lane_inds = ((nonzerox>=win_xright_low) & (nonzerox<win_xright_high)).nonzero()[0]
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    #return result, left_fitx, right_fitx
    return out_img, left_fit, right_fit, left_fitx, right_fitx, ploty


## Measurement functions ##
def measure_curvature_real(y_eval, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    left_a = left_fit_cr[0]*(xm_per_pix/ym_per_pix**2)
    left_b = left_fit_cr[1]*(xm_per_pix/ym_per_pix)
    
    right_a = right_fit_cr[0]*(xm_per_pix/ym_per_pix**2)
    right_b = right_fit_cr[1]*(xm_per_pix/ym_per_pix)
    
    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_a*y_eval + left_b)**2)**(3/2))/(np.abs(2*left_a))  ## Implement the calculation of the left line here
    right_curverad = ((1 + (2*right_a*y_eval + right_b)**2)**(3/2))/(np.abs(2*right_a)) ## Implement the calculation of the right line here
    
    
    return left_curverad, right_curverad

def measure_rel_vehicle_position(image_shape, left_fit, right_fit):
    '''
    Calculates the position of the vehicle relative to the lane center. Positive value means vehicle is left to the lane center
    '''
    
    # Calculate x position of the lanes at the bottom of the image.
    left_fitx = (left_fit[0]*image_shape[0]**2) + (left_fit[1]*image_shape[0]) + left_fit[2]
    right_fitx = (right_fit[0]*image_shape[0]**2) + (right_fit[1]*image_shape[0]) + right_fit[2]
    
    # Vehicle and center lane position
    vehicle_position = np.int(image_shape[1]//2)
    center_lane = left_fitx + (right_fitx - left_fitx)/2
    
    ## Relative position of the vehicle relative to the lane center. If positive, vehicle is left to the lane center
    rel_vehicle_position = xm_per_pix*(center_lane - vehicle_position)
    
    return np.round(rel_vehicle_position,2)

## Ploting and saving images functions ##
def save_lane_lines_image(output_path, image_name, image):

    plt.imshow(image)
    plt.title('Lane Lines Image', fontsize=20) 
    plt.savefig(os.path.join(output_path, "lane_lines_" + os.path.basename(image_name))) 
    
    plt.close()

def save_pipeline_image(output_path, image_name, pipeline_result, radius_curvature, rel_vehicle_position):

    plt.imshow(pipeline_result)
    plt.title('Pipeline result Image', fontsize=20) 

    plt.text(0, 100,"Radius of the Curvature = {0}m".format(radius_curvature), color='r', fontsize=15)

    if rel_vehicle_position > 0:
        plt.text(0, 200,"Vehicle is {0}m left of the center".format(rel_vehicle_position), color='r', fontsize=15)
    else:    
        plt.text(0, 200,"Vehicle is {0}m right of the center".format(-rel_vehicle_position), color='r', fontsize=15)

 
    plt.savefig(os.path.join(output_path, "pipeline_result_" + os.path.basename(image_name))) 
    
    plt.close()
    
def save_warped_images(output_path, image_name, original_image, warped):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
   
    ax1.imshow(original_image)
    ax1.plot([220,1100], [700, 700], 'r-')
    ax1.plot([690,1100], [450, 700], 'r-')
    ax1.plot([590,220], [450, 700], 'r-')
    ax1.plot([590,690], [450, 450], 'r-')
    ax1.set_title('Undistorted Image with source points', fontsize=30)

    
    ax2.imshow(warped)
    ax2.set_title('Warped Image with destination points', fontsize=30)
    plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0, wspace = 0.1)
    ax2.plot([350,950], [warped.shape[0]-1, warped.shape[0]-1], 'r-')
    ax2.plot([950,950], [0, warped.shape[0]-1], 'r-')
    ax2.plot([350,350], [0, warped.shape[0]-1], 'r-')
    ax2.plot([350,950], [0, 0], 'r-')
    plt.savefig(os.path.join(output_path, "warped_" + os.path.basename(image_name)))  
    
    plt.close()
