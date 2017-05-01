##Vehicle Report

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/original.png
[image4]: ./output_images/sw1.png
[image5]: ./output_images/sw2.png
[image6]: ./output_images/sw3.png
[image7]: ./output_images/heatmap.png
[image8]: ./output_images/detection.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #7 through #24 of the file called `src/feature_extraction.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and following settting is what I finally used:

spatial_feat = True
hist_feat = True
hog_feat = True
colorspace = 'YCrCb'
orient = 9 
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial = 32
histbin = 32

This feature setting gives me beat test accuracy 0.9889.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using by sklearn LinearSVC function with default setting and get a test accuracy of 0.9889. 
The training dataset is about 90% of the whole dataset. The GTI cars dataset are splitted manully to overcome the bad effect causing by time-series data. Model training code can be found in file src/train_svm.ipynb

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The slide window function can be found in src/windows.ipynb and visualization code can be found in cell #6 of src/visualization.ipynb.

The overlay of windows are all set to be 75% corresponding to the 2 cell hog features.

And three scales windows are used to serach cars:

Scale1: x_start_stop = [0, 1280], y_start_stop = [380, 500], window_size = [96, 96]

![alt text][image4]

Scale2: x_start_stop = [0, 1280], y_start_stop = [400, 600], window_size = [128, 128]

![alt text][image5]

Scale3: x_start_stop = [0, 1280], y_start_stop = [500, 660], window_size = [160, 160]

![alt text][image6]

Totally, we search for 240 windows for cars.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

The heatmap of the detection:

![alt text][image7]

The final car detection with bounding box:

![alt text][image8]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes. 

A deque with max size limit is used to store previous frame detection heat map. Sum of all previouse queued heatmap is thresholded to filter false detections. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

In order to speed up the detection processing, a mask based on previous frames' heatmap is used to reduce numbers of window to predict. Only the windows have ovelay with the mask will be fed into SVC. Code is in #123-124 of src/detection_pipeline.py

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The speed for this hog+linearSVM method is about 5 frames per seconds on my machine. The features number for training is 8460 which is not very large (larger features are tested but gives simliar accuracy). In order to make the detections run realtime, we can try with SSD which should be musch faster. Also we can feed in more data to improve detector accuracy and robustness.

