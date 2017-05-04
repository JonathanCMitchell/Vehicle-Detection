import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import settings
from scipy.ndimage.measurements import label
from helpers import convert_color, \
    get_hog_features, \
    bin_spatial, \
    color_hist, \
    draw_centroids




# Load data from pickle file
pickle_data = pickle.load(open("svc_pickle.p", "rb"))
svc = pickle_data['svc']
X_scaler = pickle_data['X_scaler']
orient = pickle_data['orient']
pix_per_cell = pickle_data['pix_per_cell']
cell_per_block = pickle_data['cell_per_block']
spatial_size = pickle_data['spatial_size']
hist_bins = pickle_data['hist_bins']

def draw_labeled_boxes(img, labels, color = (0, 0, 255), thick = 6):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image
    return img




class Car_Finder():
    def __init__(self, ystart = 400, ystop = 656, scale = 1.5):
        self.ystart = ystart
        self.ystop = ystop
        self.scale = scale
        self.threshold = 10
        self.smooth_factor = 20
        self.heatmaps = []
        self.recent_detections = []
        self.count = 0
        self.heat = np.zeros((settings.IMG_HEIGHT, settings.IMG_WIDTH), dtype = np.float32) # maybe chance dtype


    def process_image(self, img):
        self.count += 1
        centroid_rectangles = self.get_centroid_rectangles(img)

        draw_img_centroids = draw_centroids(img, centroid_rectangles)
        # labels = label(heat)
        # draw_img = draw_labeled_boxes(np.copy(img), labels)
        return draw_img_centroids

    def get_centroid_rectangles(self, img):
        # TODO: Reset heat map each time
        self.reset_heatmaps()
        centroid_rectangles = []
        detections = self.find_cars(img, self.ystart, self.ystop, self.scale, svc, X_scaler, orient, pix_per_cell,
                                    cell_per_block,
                                    spatial_size,
                                    hist_bins)
        self.update_heatmap(detections)
        self.heatmaps.append(self.heat)

        # instead of self.heat use average of last 20 heat maps
        if len(self.heatmaps) > self.smooth_factor:
            heat = np.mean(self.heatmaps[-self.smooth_factor:], axis = 0).astype(np.uint8)
        else:
            heat = self.heat.astype(np.uint8)


        binary = self.apply_threshold(heat, self.threshold)

        _, contours, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rect = cv2.boundingRect(contour)
            if rect[2] < 50 or rect[3] < 50: continue
            x, y, w, h = rect
            centroid_rectangles.append([x, y, x+w, y+h])
        # Now heatmap is binary so we apply contours
        return centroid_rectangles

    def reset_heatmaps(self):
        # Reset heatmaps every 30 frames (20 frames also works)
        if self.count % 20 == 0:
            self.heat = np.zeros((settings.IMG_HEIGHT, settings.IMG_WIDTH), dtype = np.float32) # maybe chance dtype

    def update_heatmap(self, detections):
        for (x1, y1, x2, y2) in detections:
            self.heat[y1:y2, x1:x2] += 5

    def apply_threshold(self, heatmap, threshold):
        # TODO: Impement next two line averaging function
        # if len(self.heatmaps) > self.smooth_factor:
        #     heatmap = np.mean(self.heatmaps[-self.smooth_factor:], axis = 0).astype(np.float32)
        # Threshold
        _, binary = cv2.threshold(heatmap, 11, 255, cv2.THRESH_BINARY)
        return binary

    def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                  hist_bins):
        """
        After training our SVM, we utilize find_cars in conjunction with our SVM predictor to locate the cars

        :param img: Image passed in as RGB as shape (720, 1280, 3)
        :param ystart: Starting point for search in y
        :param ystop: Stopping point for search in y
        :param scale: Scale to shrink or expand the search region
        :param svc: Linear Support Vector Machine Classifier which was trained on training data in model.py
        :param X_scaler: Linear SVC Scaler which was fit on the training data in model.py
        :param orient: # of orientations, specified as an integer, represents the number of orientation bins that the 
        gradient information will be split up into the histogram. Typically ~6 to 12 .
        :param pix_per_cell: # of pixels per cell passed to HOG as (pix_per_cell, pix_per_cell) tuple. 
        They are commonly chosen to be square
        :param cell_per_block: number of cells per cell_block passed to HOG as (cell_per_block, cell_per_block) tuple. 
        It specifies the local area over which the histogram counts in a given cell will be normalized.
        :param spatial_size: tuple size passed into bin_spatial in helper.py, used to resize the image and determines
         the length of the features return from bin_spatial after ravel()-ing
        :param hist_bins: number of bins inside np.histogram of a specific channel inside the image, correlates to the resolution of the histogram features, 
        each bin represents a value that has a corresponding feature, more bins = more features captured
        :return: draw_img (720, 1280, 3) RGB copy of img that has boxes drawn on it by using cv2.rectangle
        """
        draw_img = np.copy(img)
        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        # Channel extraction
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        bboxes = []
        detections = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch

                # Flatten the HOG features for each channel position
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                # Build hog_features array by stacking each channel
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                confidence = svc.decision_function(test_features)

                if test_prediction == 1 and confidence > 0.6:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    x1 = xbox_left
                    y1 = ytop_draw + ystart
                    x2 = xbox_left + win_draw
                    y2 = ytop_draw + win_draw + ystart
                    detections.append((x1, y1, x2, y2))
        return detections


