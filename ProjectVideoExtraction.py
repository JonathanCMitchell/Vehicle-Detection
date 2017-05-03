import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
import csv
from tqdm import tqdm

# Project Video './project_video.mp4'
#
with open('./project_video_data/driving.csv', 'w') as csvfile:
    fieldnames = ['image_path', 'frame']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    cap = cv2.VideoCapture('./project_video.mp4')
    cap.set(cv2.CAP_PROP_FRAME_COUNT, 1250)

    for idx in tqdm(range(1250)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        success, image = cap.read()
        if success:
            image_path = os.path.join('./project_video_data/IMG', str(idx) + '.jpg')
#             img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, image)

            writer.writerow({'image_path': image_path, 'frame': idx})
print('done!')