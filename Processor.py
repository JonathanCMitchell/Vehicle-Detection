import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from find_cars import Car_Finder
import pandas as pd
import cv2
from tqdm import tqdm
import time

df = pd.read_csv('./project_video_data/driving.csv')
cf = Car_Finder()
#
# start = 300
# stop = 355
# for i in tqdm(range(start, stop)):
#     impath = df.iloc[[i]]['image_path'].values[0]
#     img = mpimg.imread(impath)
#     image, heatmap = cf.process_image(img)
#     cv2.imwrite('./results/images/' + str(i) + 'image' + '.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     cv2.imwrite('./results/heatmaps/'+ str(i) + 'heatmap' + '.jpg', heatmap)

## img = mpimg.imread('./test_images/test1.jpg')
#
# test_images = glob.glob("./test_images/*.jpg")
# print('test_images:', test_images)
# for image in test_images:
#     img = mpimg.imread(image)
#     out_img, heatmap = cf.process_image(img)
#     plt.subplot(211)
#     plt.imshow(out_img)
#     plt.subplot(212)
#     plt.imshow(heatmap)
# plt.show()
    # cv2.imwrite('./results/' + str(image) + 'processed.jpg', out_image)






## MOVIEPY
from moviepy.editor import VideoFileClip

t = time.time()
test_output = 'project_video_output_X8.mp4'
clip1 = VideoFileClip("project_video.mp4")
cf = Car_Finder()
white_clip = clip1.fl_image(cf.process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(test_output, audio=False)
t2 = time.time()
print('time: ', (t2 - t) / 60)
