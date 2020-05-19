## Writeup 
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration/undist_calibration1.png "Undistorted Chessboard Image"
[image2]: ./output_images/test_images/undistorted_straight_lines1.jpg "Undistorted Image"
[image3]: ./output_images/test_images/binary_straight_lines1.jpg "Binary Example"
[image4]: ./output_images/test_images/undistorted_straight_lines1.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/test_images/pipeline_result_straight_lines1.jpg"Output"
[video1]: ./project_video.mp4 "Video"

## Approach

To accomplish with the project goals, presented above, I divided the project development in four phases:
1 - Camera Calibration. Calculate the camera matrix and distortion coefficients using the chessboard images provided in "camera_cal" folder.
2 - Build a Lane finding Pipeline to a single image
3 - Build a Lane finding Pipeline to a video
4 - Reinforce the video Pipeline using the provided challenge videos 


### Here I will consider the 4 phases individually and describe how I addressed each phase in my implementation.  

---

### 1. Camera Calibration

#### Here is a description of how I computed the camera matrix and distortion coefficients. An example of a distortion corrected calibration image is provided.

The code for this step is contained in the third code cell of the IPython notebook located in "main.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the image. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Furthermore I am counting (9,6) corners on each image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image, using the  `cv2.findChessboardCorners()` function over the gray scale of the test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

After completing the search of the chessboard corners of all test images, I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function to verify that all process is correct. An example of the result obtained is: 

![alt text][image1]

To avoid running every time this cell, I stored the camera calibration and distortion coefficients into the file `camera_coeff.pkl` so that next time I just have to load the file.

### 2. Build a Lane finding Pipeline to a single image

In this phase, the following steps were performed, first applied to the straight lines images then to the other test images:
* Apply a distortion correction to the images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The pipeline function calls `process_image` (6th code cell of the IPython notebook located in "main.ipynb") which was built step by step. Its arguments are:
* image - the image object (read from mpimg.imread) 
* fname - the image name for saving purposes
* mtx, dist - camera matrix and distortion coefficients determined on phase 1
* M, M_inv - transform matrix M and inverted transform matrix M_inv. These matrixes were first determined on step C but then removed from this pipeline and included here as argument. Now they are determined on 5th code cell of the IPython notebook located in "main.ipynb".

#### A. Apply a distortion correction to the images.

The distortion correction to the images is done by function `undistort_image` defined on file `helper_functions.py`. This function uses `cv2.undistort`function to perform the distortion correction using the camera matrix and distortion coefficients calculated on phase 1.

A result of this step is:

![alt text][image2]

#### B. Use color transforms, gradients, etc., to create a thresholded binary image.

The approach to this step was to tune separately the threshold values of all binary images using the color transform or gradients and then combined all of then. The functions used on this process are declared on file `helper_functions.py` (`abs_sobel_thresh`, `mag_thresh`, `dir_threshold`, `hls_select`) 
The final result is a combination of S-channel binary image plus the x and y gradients plus magnitude and direction gradients. 
The result of this combination is the binary image called `binary_image` determined as:
`binary_image[(colors_binary == 1) | ((mag_binary == 1) & (dir_binary == 1)) | ((gradx == 1) & (grady == 1)) ] = 1`
   
Here's an example of my output for this step.  

![alt text][image3]

#### C. Apply a perspective transform to rectify binary image ("birds-eye view").

The code used to obtain the transform matrix M and M_inv needed to perform perspective transform to the image is presented in the 5th code cell of the IPython notebook located in "main.ipynb".  Using the test image `straight_lines1` as the base image, I explored the image usind a image editor and after several attemps, I defined the source (`src`) and destination (`dst`) points as:

```
src = np.float32([[195, 720],[1125, 720],[578, 460],[705, 460]])
dst = np.float32([[350, 720],[950, 720],[350,0],[950,0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto the test image `straight_lines1` and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

On the pipeline, the transform matrix M and M_inv are passed as argument to the function `process_image` and the perspective transform is done calling the function `cv2.warpPerspective`.

#### D.Detect lane pixels and fit to find the lane boundary.

This step is perfomed on function `fit_polynomial` which take as argument the binary_warped image and return 6 objects:
* out_img - warped image with the polynomial lines for left and right lane (yellow) and the x,y points used to fit that polynomial lines (red for left lane and blue for right lane) 
* left_fit, right_fit - polynomial coefficients for left and right polynomial lines, respectively
* left_fitx, right_fitx - x points of the left and right polynomial lines, respectively
* ploty - y points of the left and right polynomial lines (these values are the same for left and right lines)

On this function, the first thing done is to find the binary image pixels with coordinates (x,y) needed to fit a 2nd order polynomial line. This step is done by the function `find_lane_pixels` which use the sliding window method and works as follows:

* Take a histogram of the bottom half of the image and find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines (leftx_base and rightx_base).
* Set up windows and window hyperparameters.
* Calculate the nonzero x and y pixels using the function nonzero().
* Loop through each window and find the boundaries of the current window. This is based on a combination of the current window's starting point (leftx_current and rightx_current), as well as the hyperparameter margin.
* I use cv2.rectangle to draw these window boundaries onto our visualization image out_img. 
* Knowing the boundaries of our window, I find out which activated pixels, on both x and y coordinates, actually fall into the window.
* Append the pixels index to the lists left_lane_inds and right_lane_inds.
* If the number of activated pixels are greater than your hyperparameter minpix, re-center our window (based on the mean position of these pixels.
* In the end of the loop, I extract left and right line pixel positions.

After getting the left and right line pixel positions, in function `fit_polynomial`, I Fit a second order polynomial to each using `np.polyfit` and then generate x and y values of that polynomial line to add the polynomial line onto the image.

![alt text][image5]

#### E. Determine the curvature of the lane and vehicle position with respect to center.

After detecting the left and right lanes and fit a second order polynomial to each, I determine the curvature of the lane on function `measure_curvature_real` where I first convert the polynomial coefficients in pixel units to meter using the conversions:
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/600 # meters per pixel in x dimension

and the formulas: a_m = a_p*(xm_per_pix/ym_per_pix**2); b_m = b_p * (xm_per_pix/ym_per_pix)

After that I average the two polynomial coefficients and calculate the curvature of the lane as:
R = ((1 + (2*a*y*ym_per_pix + b)**2)**(3/2))/(np.abs(2*a))

The vehicle position with respect to center of the lane is calculated on function `measure_rel_vehicle_position` where are calculated the x_position of the lane lines at the bottom of the image. 

The the center of the lane is determined as the point in the middle of that two x_position points and the vehicle position is on the midle of the image width. The relative position is then calculated as the diference of that two values. 

The conversion to meter unit is also done using the conversion xm_per_pix.

**Results obtained**:

The curvature values for the test images were calculated around 700 - 1200 m, except for the straight lines were the curvature values were much higher > 7000 m.

The vehicle position for the test images were calculated around 0.04m to 0.35m

#### F. Warp the detected lane boundaries back onto the original image.

With the lane boundaries detected, it is created a binary image to draw the lane lines for output display. But first it is necessary to create the lane lines and then warp back to the original image space. So the next steps are done:
* Create an image to draw the lines on
* Recast the x and y points into usable format for `cv2.fillPoly`
* Draw the lanes onto the warped blank image
* Warp the blank back to original image space using inverse perspective matrix (Minv) with the function `cv2.warpPerspective`

#### G. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position..

To display the original image with the lane boundaries and numerical estimation of lane curvature and vehicle position, it is necessary to combine the unwarped binary image defined on previous step (F) with the original image. This combination is done using the function `cv2.addWeighted`.

After that two texts are added on function `save_pipeline_image`. These two text have the information about numerical estimation of lane curvature and vehicle position.


#### Result of the Lane finding Pipeline

I implemented this pipeline in the 6th code cell of the IPython notebook located in "main.ipynb" though the function `process_image`. This function is called for all test images on 7th code cell of the IPython notebook located in "main.ipynb" and the results are stored on folder `/output_images/test_images`. 
An example of pipeline result on a test image is:

![alt text][image6]

---

### 3. Build a Lane finding Pipeline to a video



#### A. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
