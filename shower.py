import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# heatmap = mpimg.imread('./nh_results/heatmaps/241heatmap.jpg')
averaged_heatmap = mpimg.imread('./final_results/averaged_heatmaps/800averaged_heatmap.jpg')
thresholded_heatmap = mpimg.imread('./final_results/thresholded_heatmaps/800thresholded_heatmap.jpg')
final_result = mpimg.imread('./final_results/images/800image.jpg')
original_image = mpimg.imread('./project_video_data/IMG/800.jpg')

plt.imshow(original_image)

threshold = 0.6
_, binary = cv2.threshold(averaged_heatmap, threshold, 255, cv2.THRESH_BINARY)
#
_, contours, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # plt.imshow(binary, cmap='hot')
print('contours are :', contours)
for contour in contours:
    rect = cv2.boundingRect(contour)
    plt.plot(rect[0], rect[1], 'c^')
    plt.plot(rect[0] + rect[2],rect[1] + rect[3],'g^')
    plt.plot(rect[0] + rect[2],rect[1], 'b^')
    plt.plot(rect[0], rect[1] + rect[3], 'r^')
    plt.title('original image with SVM predictions')
    plt.axis('off')
    print('rect: ', rect)

plt.show()
#
# for contour in contours:
#     rect = cv2.boundingRect(contour)
#     if rect[2] < 50 or rect[3] < 50: continue
#     x, y, w, h = rect
#     centroid_rectangles.append([x, y, x + w, y + h])
# # Now heatmap is binary so we apply contours
# return centroid_rectangles