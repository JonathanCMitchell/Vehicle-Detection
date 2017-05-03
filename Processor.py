import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from find_cars import Car_Finder
import cv2

img = mpimg.imread('./test_images/test1.jpg')
cf = Car_Finder()

test_images = glob.glob("./test_images/*.jpg")
print('test_images:', test_images)
for image in test_images:
    img = mpimg.imread(image)
    out_img, heatmap = cf.process_image(img)
    plt.subplot(211)
    plt.imshow(out_img)
    plt.subplot(212)
    plt.imshow(heatmap)
plt.show()
    # cv2.imwrite('./results/' + str(image) + 'processed.jpg', out_image)






## MOVIEPY
# from moviepy.editor import VideoFileClip
#
# test_output = 'project_video_output2.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# cf = Car_Finder()
# white_clip = clip1.fl_image(cf.process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(test_output, audio=False)