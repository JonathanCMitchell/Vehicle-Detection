import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import settings

from helpers import convert_color, \
    get_hog_features, \
    bin_spatial, \
    color_hist, \
    draw_boxes



img = mpimg.imread('./test_images/test1.jpg')



# Load data from pickle file
pickle_data = pickle.load(open("svc_pickle.p", "rb"))
svc = pickle_data['svc']
X_scaler = pickle_data['X_scaler']
orient = pickle_data['orient']
pix_per_cell = pickle_data['pix_per_cell']
cell_per_block = pickle_data['cell_per_block']
spatial_size = pickle_data['spatial_size']
hist_bins = pickle_data['hist_bins']

#
#
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
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
    # img = img.astype(np.float32) / 255

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

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bboxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    draw_img = draw_boxes(draw_img, bboxes)
    return draw_img


ystart = 400
ystop = 656
scale = 1.5

out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                    hist_bins)

# HOG Subsample search region
# y_vals = np.linspace(ystart, ystart, out_img.shape[1] - 1 )
# x_vals = np.linspace(0, out_img.shape[1], out_img.shape[1] - 1)
# y_vals2 = np.linspace(ystop, ystop, out_img.shape[1] - 1)
# plt.plot(x_vals, y_vals, 's-')
# plt.plot(x_vals, y_vals2, 'bs-')
# plt.title('HOG Subsample search region')


plt.imshow(out_img)
plt.title('HOG subsampling method on test1')
plt.show()