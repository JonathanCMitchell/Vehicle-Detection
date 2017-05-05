import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
import settings
from tqdm import tqdm
pickle_data = pickle.load(open("svc_pickle.p", "rb"))
svc = pickle_data['svc']
X_scaler = pickle_data['X_scaler']
orient = pickle_data['orient']
pix_per_cell = pickle_data['pix_per_cell']
cell_per_block = pickle_data['cell_per_block']
spatial_size = pickle_data['spatial_size']
hist_bins = pickle_data['hist_bins']


from helpers import get_hog_features

original_image = mpimg.imread('./project_video_data/IMG/800.jpg')
img = cv2.cvtColor(original_image, cv2.COLOR_RGB2YCrCb)

features, hog_image_Y = hog(img[:,:,0], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=True)
features, hog_image_Cr = hog(img[:,:,0], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=True)
features, hog_image_Cb = hog(img[:,:,0], orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=True)

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax1.title.set_text('YCrCb')
ax1.imshow(img)
plt.axis('off')

ax2 = plt.subplot2grid((3, 3), (2, 1))
ax2.title.set_text('HOG ALL channel')
ax2.imshow(np.dstack((hog_image_Y, hog_image_Cr, hog_image_Cb)))
plt.axis('off')

ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax3.title.set_text('HOG Cr channel')
ax3.imshow(hog_image_Cr, cmap='gray')
plt.axis('off')

ax4 = plt.subplot2grid((3, 3), (2, 0))
ax4.title.set_text('HOG Cb channel')
ax4.imshow(hog_image_Cb, cmap='gray')
plt.axis('off')

ax5 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax5.title.set_text('HOG Y channel')
ax5.imshow(hog_image_Y, cmap='gray')
plt.axis('off')

plt.show()
# plt.subplot(311)
# plt.title('original image in YCrCb')
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(321)
# plt.title('hog image channel Y')
# plt.imshow(hog_image_Y, cmap='gray')
# plt.axis('off')
#
# plt.subplot(322)
# plt.title('hog image channel Cr')
# plt.imshow(hog_image_Cr, cmap='gray')
# plt.axis('off')
#
# plt.subplot(21)
# plt.title('hog image channel Cb')
# plt.imshow(hog_image_Cb, cmap='gray')
# plt.axis('off')
#
# plt.subplot(522)
# plt.title('hog image ALL channels')
# plt.imshow(np.dstack((hog_image_Y, hog_image_Cr, hog_image_Cb)))
# plt.axis('off')
#
# plt.show()
# # heatmap = mpimg.imread('./nh_results/heatmaps/241heatmap.jpg')
# averaged_heatmap = mpimg.imread('./final_results/averaged_heatmaps/800averaged_heatmap.jpg')
# thresholded_heatmap = mpimg.imread('./final_results/thresholded_heatmaps/800thresholded_heatmap.jpg')
# final_result = mpimg.imread('./final_results/images/800image.jpg')
# original_image = mpimg.imread('./project_video_data/IMG/800.jpg')
#
# plt.imshow(original_image)
#
# threshold = 0.6
# _, binary = cv2.threshold(averaged_heatmap, threshold, 255, cv2.THRESH_BINARY)
# #
# _, contours, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #
# # # plt.imshow(binary, cmap='hot')
# print('contours are :', contours)
# for contour in contours:
#     rect = cv2.boundingRect(contour)
#     plt.plot(rect[0], rect[1], 'c^')
#     plt.plot(rect[0] + rect[2],rect[1] + rect[3],'g^')
#     plt.plot(rect[0] + rect[2],rect[1], 'b^')
#     plt.plot(rect[0], rect[1] + rect[3], 'r^')
#     plt.title('original image with SVM predictions')
#     plt.axis('off')
#     print('rect: ', rect)
#
# plt.show()
# #
# # for contour in contours:
# #     rect = cv2.boundingRect(contour)
# #     if rect[2] < 50 or rect[3] < 50: continue
# #     x, y, w, h = rect
# #     centroid_rectangles.append([x, y, x + w, y + h])
# # # Now heatmap is binary so we apply contours
# # return centroid_rectangles