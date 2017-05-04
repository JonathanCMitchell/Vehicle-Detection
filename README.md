# Vehicle Detection

## Important files
* model.py (builds SVM classifier and trains on training data)
* Processor.py (overlaying processor that runs the pipeline on a video stream input)
* settings.py (settings for tuned parameters)
* find_cars.py (contains Car_Finder class that applies the model to detect cars and draws rectangular boxes on images)

## Training data:
* Our training dataset consists of 17760 images. 8792 non-vehicles and 8968 vehicles. We split our training and validation sets with 80% 20% respectively. So our training dataset was ~14208 images.


### Process:
#### 1) Training data (model.py)
* Grab the training data and extract image paths and store them in the model class
* Train a linear Support Vector Machine (lines 54 - 57)
* Split the training data using `train_test_split`. Keep in mind that we are using training data as validation data, so there is some overfitting there. On a time series analysis it would be more robust to check a time-range and split validation data so that it has distinct times from the training data
* Normalize the training data using mean - std normalization, using scikit's `StandardScaler` (line 49)
* Train the model on the data
* Save the LinearSVC (Linear Support Vector Machine) to a pickle file as well as other parameters and move to step 2 (line 79)

#### 2) Car detection (find_cars.py)
##### In Car_Finder.find_cars
* Grid out a section of the image (height from 400 to 656) and all width.  (line 138)
* Extract the HOG features for the entire section of the image
* ![HOG_subsample](https://github.com/JonathanCMitchell/Vehicle-Detection/blob/master/output_images/HOG_subsample_search_region.png)
* Above: You can see the region we are using for our sub image between the light and dark blue lines
* Scale the extracted section by a `scale` parameter (line 142)
* Extract each channel from the scaled image
* Calculate the number of blocks in x and y
* Define a search window
* Create a step size `(cells_per_window / step_size) = % overlap`
* Discover how many vertical and horizontal steps you will have
* Calculate [HOG](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) features for each channel lines (161 - 163)
* Consider the scaled image to be in position space, not image space. We treat sections of the image as a grid in terms of whole integer values instead of in pixel values
# TODO: Show grid image (take picture on phone for it)
* We will move from grid space back into image space later on don't worry
* For now, consider xpos and ypos to be grid positions (from left to right) (lines 168-169)
* Iterate through the grid (in y then x) lines (lines 166-168)
* Grab a subsample of the hog features corresponding to the current cell
* stack the features
* go from grid space to pixel space (lines 179-180) now xleft and ytop are in image space
* extract the image patch (as defined by the boundaries of xleft and ytop and the window size)
* get the spatial and histogram features using `spatial_features` and `hist_features` which are defined in helpers.py
* Normalize the features using `X_scaler` which is from model.py
* Stack the features (line 191)
* Compute the prediction from the Support Vector Machine and the confidence
* If the prediction is 1 and the decision function confidence is > 0.6 then we have detected a car
* Rescale `xbox_left` and `ytop_draw` to go from our current image space (which is scaled) to real image space by multiplying it by the scaling factor `scale`
* Use the drawing window dimensions to build (x1, y1, x2, y2) detection coordinates which will help us build a heatmap for the detected cars location
##### In Car_Finder.get_centroid_rectangles
* Take in the output from `Car_Finder.find_cars` (which are the detection coordinates above)
* Reset the heatmap if 20 frames have passed since our last reset
* Take in the detection coordinates and update the heatmap by adding 5 to each value within the heatmap's bounding box
* ![hog_subsampling_on_test1](https://github.com/JonathanCMitchell/Vehicle-Detection/blob/master/output_images/HOG_subsampling_on_test1.png)
* Above, as you can see we have more than one bounding box. Therefore we need to apply a heatmap in order to determine an accurate region for the vehicle and only draw one box

* ![image_heatmap_sidebyside](https://github.com/JonathanCMitchell/Vehicle-Detection/blob/master/output_images/processed_test_img1_and_heatmap.png)
* As you can see, sometimes we get detections that are false positives, in order to remove these false positives we apply a thresholding function
* Remove the values where the heatmap's values are < 20. So it takes ~4 heat maps to pass through the thresholder
* Before we threshold, we take an average of the last 20 heat maps if 20 heat maps have been saved, then we insert this map into our thresholder
* Averaging allows us to rule out bad values and creates a smoother transition between frames
* Then we find the contours for the binary image, (which are basically the large shapes created from the heatmap)
* Then we create bounding boxes from these contours
* ![HOG_subsampling_on_test4](https://github.com/JonathanCMitchell/Vehicle-Detection/blob/master/output_images/HOG_subsampling_on_test4.png)
* Above you can see that we have some false positives in the opposing lane, therefore we will rule out any boxes that occur at width < 50 because this area corresponds to the opposing highway. We do this on line 91 
* Grab the coordinates of the bounding box and append them to `centroid_rectangles` which we will pass to our `draw_centroids` helper function
##### Draw Centroids (in helpers.py)
* Get the centroid coordinates and the original image and draw a rectangle with the coordinates given
* Return this image.
* Note: I got the idea to find the contours from [Kyle Stewart-Frantz](https://github.com/kylesf)


#### Helper functions (in helpers.py)
* <strong>Extract_features</strong> (lines 61 - 112). This function takes in list of image paths, reads the images, then extracts the spatial, histogram, and hog features of those images and saves them. This occurs in our preprocessing pipeline when we are training the model. Our SVM does not extract features itself, so we have to extract them from images, similar to how a multi-layer perceptron works, in contrast with how CNN's work.
* We extract features using three helper functions:
* <strong> bin_spatial </strong>: This function resizes our image to a desired size and then flattens it, giving us the pixel values in a flattened row vector
* <strong> color_hist</strong>: This function computes the histogram for each color channel and then concatenates them together
* <strong> get_hog_features</strong>: This function computes the histogram of oriented gradient and returns a 1-D array of the HOG features
* Then inside Extract_features we grab all of these features for each image and then add them to our `features` list. So `features` contains the features for the entire data set where a `file_feature` contains the features for one image
THE END

#### Details (Parameter selection) (tuning params.ods)
* ![tuning_params](https://github.com/JonathanCMitchell/Vehicle-Detection/blob/master/output_images/tuning_params.png)
* Above: I tried these different parameters and tested the SVM's predictions on a single image. I chose the YCrCb color space because it gave me the best accuracy at training time
* ![YCrCb_allChannel](https://github.com/JonathanCMitchell/Vehicle-Detection/blob/master/output_images/YCrCb_detection_ALL_Channel.png)
* Above: Result of training using YCrCb color space
* This gives us the best result with an accuracy of 99.5. Don't be fooled by the 1.0 in the grid of parameter tests, that was only sampled for one image
* ![LUV_detection_L_channel](https://github.com/JonathanCMitchell/Vehicle-Detection/blob/master/output_images/LUV_detection_L_channel.png)
* Above: Result of training using L channel in LUV color space
* ![LUV_detection_V_channel](https://github.com/JonathanCMitchell/Vehicle-Detection/blob/master/output_images/LUV_detection_V_channel.png)
* Above: Result of training using V channel in LUV color space 



### Videos
#### reset heatmap every 30 frames
<a href="http://www.youtube.com/embed/-AV2jysSHcQ
" target="_blank"><img src="http://img.youtube.com/vi/-AV2jysSHcQ/0.jpg" 
alt="Watch Video Here" width="480" height="180" border="10" /></a>

#### reset heatmap every 20 frames
<a href="http://www.youtube.com/embed/GdUVhW7qM3w
" target="_blank"><img src="http://img.youtube.com/vi/GdUVhW7qM3w/0.jpg" 
alt="Watch Video Here" width="480" height="180" border="10" /></a>

##### Reflection: 

#### Twitter: [@jonathancmitch](https://twitter.com/jonathancmitch)
#### Linkedin: [https://www.linkedin.com/in/jonathancmitchell](https://twitter.com/jonathancmitch)
#### Github: [github.com/jonathancmitchell](github.com/jonathancmitchell)
#### Medium: [https://medium.com/@jmitchell1991](https://medium.com/@jmitchell1991)

#### Tools used
* [Numpy](http://www.numpy.org/)
* [OpenCV3](http://pandas.pydata.org/)
* [Python](https://www.python.org/)
* [Pandas](http://pandas.pydata.org/)
* [Matplotlib](http://matplotlib.org/api/pyplot_api.html)
* [SciKit-Learn](http://scikit-learn.org/)
