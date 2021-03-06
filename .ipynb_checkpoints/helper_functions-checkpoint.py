import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meters per pixel in x dimension

def undistort_image(image, mtx, dist):
    """Undistort the image based on camera coefficients"""
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return  undist


## Thresholded binary image functions ##
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Create a thresholded binary image based on directional gradient"""
    # Calculate directional gradient
    # Apply threshold
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x or y separately
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
    """Create a thresholded binary image based on gradient magnitude"""
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
    """Create a thresholded binary image based on gradient direction"""
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

def hls_select(img, s_thresh=(170, 255), l_thresh=(170, 255)):
    """Create a thresholded binary image based on s channel and l channel"""
    
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l = hls[:,:,1]
    s = hls[:,:,2]
      
    # 2) Apply a threshold to the S channel
    s_binary_output = np.zeros_like(s)
    s_binary_output[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 1
    
    # 3) Apply a threshold to the L channel
    l_binary_output = np.zeros_like(l)
    l_binary_output[(l >= l_thresh[0]) & (l <= l_thresh[1])] = 1
    
    # 3) Merge the two binary images
    binary_output = np.zeros_like(s)
    binary_output[(s_binary_output == 1) | (l_binary_output == 1)] = 1
    
    return binary_output

def hsv_select(img, s_thresh=(100, 255), v_thresh=(200, 255), vs_thresh=(180, 255), clahe = None):
    """Create a thresholded binary image based on s channel and v channel"""
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # 1) Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = hsv[:,:,1]
    v = hsv[:,:,2]
     
    # 2) Apply adaptive histogram equalization to remove brightness
    v = clahe.apply(v)
    s = clahe.apply(s)

    # 3) Apply a threshold to the V channel
    v_binary_output = np.zeros_like(v)
    v_binary_output[(v >= v_thresh[0]) & (v <= v_thresh[1])] = 1

    # 4) Apply a threshold to the S channel
    s_binary_output = np.zeros_like(s)
    s_binary_output[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 1
    
    # 4) Apply a second threshold to the V channel
    vs_binary_output = np.zeros_like(v)
    vs_binary_output[(v >= vs_thresh[0]) & (v <= vs_thresh[1])] = 1
    
    # 5) Merge the three binary images
    binary_output = np.zeros_like(s)
    binary_output[((s_binary_output == 1) & (vs_binary_output == 1)) | (v_binary_output == 1)] = 1
    
    return binary_output

## Finding Lane functions ##
def find_lane_pixels(binary_warped):
    """
    Find the x and y pixels for the left and right lane lines.
    Sliding windows will be used around starting points, determined as the histogram peaks of the bottom half of the image
    """
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    #histogram = np.sum(binary_warped[:-binary_warped.shape[0]//10,:], axis=0)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    

    # HYPERPARAMETERS
    # Number of sliding windows
    nwindows = 12
    # Width of the windows +/- margin
    margin = 50
    # Minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows and image shape
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

    y_min = 0
    
    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        # Find the four below boundaries of the window 
        win_xleft_low = leftx_current - margin 
        win_xleft_high = leftx_current + margin 
        win_xright_low = rightx_current - margin 
        win_xright_high = rightx_current + margin  
        
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzerox>=win_xleft_low) & (nonzerox<win_xleft_high) &
        (nonzeroy>=win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox>=win_xright_low) & (nonzerox<win_xright_high) &
        (nonzeroy>=win_y_low) & (nonzeroy<win_y_high)).nonzero()[0]
      
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window 
        # (`right` or `leftx_current`) on their mean position 
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds])) 
            
        if ((leftx_current - margin ) <= 0) | ((rightx_current + margin)>= binary_warped.shape[1]):
            print("The curve crosses the lateral boundaries")
            y_min = win_y_high
            break

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print("Error :", ValueError)
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, y_min, out_img


def fit_polynomial(binary_warped):
    """
    Fit a quadratic polynomial for the left and right lane lines based on the x,y pixels which fall on sliding windows. 
    """
    
    # Find our lane pixels first
    leftx, lefty, rightx, righty, y_min, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2 )
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(y_min, binary_warped.shape[0] -1, binary_warped.shape[0] - y_min  )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, left_fitx, right_fitx, ploty

## Measurement functions ##
def measure_curvature_real(y_eval, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of the lane based on left and right polynomial functions. The curvature is returned in meters.
    '''
    
    #Convert y point in meter unit
    y_eval = y_eval*ym_per_pix
    
    order = np.max([len(left_fit_cr), len(right_fit_cr)]) - 1
    left_dy, right_dy = 0, 0
    left_ddy, right_ddy = 0, 0
    
    for i in range(order):
        left_coef = left_fit_cr[i]*(xm_per_pix/ym_per_pix**(order-i))
        left_dy += ((order-i)*left_coef*(y_eval**(order-i-1)))            
                    
        right_coef = right_fit_cr[i]*(xm_per_pix/ym_per_pix**(order-i))
        right_dy += ((order-i)*right_coef*(y_eval**(order-i-1)))
        
        if i < order-1:
            left_ddy += ((order-i-1)*(order-i)*left_coef*(y_eval**(order-i-2)))            
            right_ddy += ((order-i-1)*(order-i)*right_coef*(y_eval**(order-i-2)))
        
    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (left_dy)**2)**(3/2))/(np.abs(left_ddy))  
    right_curverad = ((1 + (right_dy)**2)**(3/2))/(np.abs(right_ddy))                  
    
    # Average the left and right curvatures
    radius_curvature = round(np.mean([left_curverad,right_curverad]),0)
    
    return radius_curvature, round(left_curverad,0), round(right_curverad,0)

def measure_rel_vehicle_position(image_shape, left_fit, right_fit):
    '''
    Calculates the position of the vehicle relative to the lane center. Positive value means vehicle is left to the lane center
    '''
    order = np.max([len(left_fit), len(right_fit)]) - 1
        
    # Calculate x position of the lanes at the bottom of the image.
    left_fitx, right_fitx = left_fit[order], right_fit[order]
    for i in range(order):
        left_fitx += (left_fit[i]*image_shape[0]**(order-i))
        right_fitx += (right_fit[i]*image_shape[0]**(order-i))
    
    # Vehicle and center lane position
    vehicle_position = np.int(image_shape[1]//2)
    center_lane = left_fitx + (right_fitx - left_fitx)/2
    
    ## Relative position of the vehicle relative to the lane center. If positive, vehicle is left to the lane center
    rel_vehicle_position = xm_per_pix*(center_lane - vehicle_position)
    
    return np.round(rel_vehicle_position,2)

## Plotting and saving images functions ##
def save_undistorted_images(output_path, image_name, original_image, undist):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
   
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=30)

    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0, wspace = 0.1)
    plt.savefig(os.path.join(output_path, "undistorted_" + os.path.basename(image_name)))  
    
    plt.close()
    
def save_binary_images(output_path, image_name, undist, binary_image):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
   
    ax1.imshow(undist)
    ax1.set_title('Undistorted Image', fontsize=30)

    ax2.imshow(binary_image, cmap='gray')
    ax2.set_title('Binary Image', fontsize=30)
    plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0, wspace = 0.1)
    plt.savefig(os.path.join(output_path, "binary_" + os.path.basename(image_name)))  
    
    plt.close()
    
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

def save_warped_images_(output_path, image_name, original_image, warped):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
   
    ax1.imshow(original_image)
    
    ax1.plot([195,1125], [warped.shape[0]-1, warped.shape[0]-1], 'r-')
    ax1.plot([705,1125], [460, warped.shape[0]-1], 'r-')
    ax1.plot([578,195], [460, warped.shape[0]-1], 'r-')
    ax1.plot([578,705], [460, 460], 'r-')
    ax1.set_title('Undistorted Image with source points', fontsize=30)

    
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('Warped Image with destination points', fontsize=30)
    plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0, wspace = 0.1)
    ax2.plot([350,950], [warped.shape[0]-1, warped.shape[0]-1], 'r-')
    ax2.plot([950,950], [0, warped.shape[0]-1], 'r-')
    ax2.plot([350,350], [0, warped.shape[0]-1], 'r-')
    ax2.plot([350,950], [0, 0], 'r-')
    plt.savefig(os.path.join(output_path, "warped_" + os.path.basename(image_name)))  
    
    plt.close()
    
def save_warped_images(output_path, image_name, original_image, warped):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
   
    ax1.imshow(original_image)
    
    ax1.plot([280,1150], [warped.shape[0]-1, warped.shape[0]-1], 'r-')
    ax1.plot([725,1150], [480, warped.shape[0]-1], 'r-')
    ax1.plot([600,280], [480, warped.shape[0]-1], 'r-')
    ax1.plot([600,725], [480, 480], 'r-')
    ax1.set_title('Undistorted Image with source points', fontsize=30)

    
    ax2.imshow(warped, cmap='gray')
    ax2.set_title('Warped Image with destination points', fontsize=30)
    plt.subplots_adjust(left=0.1, right=0.9, top=1, bottom=0, wspace = 0.1)
    ax2.plot([350,950], [warped.shape[0]-1, warped.shape[0]-1], 'r-')
    ax2.plot([950,950], [0, warped.shape[0]-1], 'r-')
    ax2.plot([350,350], [0, warped.shape[0]-1], 'r-')
    ax2.plot([350,950], [0, 0], 'r-')
    plt.savefig(os.path.join(output_path, "warped_" + os.path.basename(image_name)))  
    
    plt.close()
