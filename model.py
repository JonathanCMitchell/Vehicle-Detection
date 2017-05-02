import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from helpers import *
from sklearn.model_selection import train_test_split
import pickle
import settings

# Read in cars and notcars
cars = []
notcars = []
images = glob.glob('./data/small_dataset/*/*/*.jpeg')
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

class model():
    def __init__(self):
        self.features = None
        self.X_scaler = None
        self.svc = None

    def predict(self):
        pass
    def get_features(self):
        car_features = extract_features(cars, color_space=settings.color_space,
                                spatial_size=settings.spatial_size, hist_bins=settings.hist_bins,
                                orient=settings.orient, pix_per_cell=settings.pix_per_cell,
                                cell_per_block=settings.cell_per_block,
                                hog_channel=settings.hog_channel, spatial_feat=settings.spatial_feat,
                                hist_feat=settings.hist_feat, hog_feat=settings.hog_feat)
        notcar_features = extract_features(notcars, color_space=settings.color_space,
                                   spatial_size=settings.spatial_size, hist_bins=settings.hist_bins,
                                   orient=settings.orient, pix_per_cell=settings.pix_per_cell,
                                   cell_per_block=settings.cell_per_block,
                                   hog_channel=settings.hog_channel, spatial_feat=settings.spatial_feat,
                                   hist_feat=settings.hist_feat, hog_feat=settings.hog_feat)
        features = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Preprocess data (scale)
        features_scaled = self.scale_data(features)
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # split up data into randomized training and test set
        rand_state = np.random.randint(0, 100)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features_scaled, labels, test_size = 0.2, random_state = rand_state)

    def scale_data(self, data):
        self.X_scaler = StandardScaler().fit(data)
        scaled_X = self.X_scaler.transform(data)
        return scaled_X

    def train(self):
        self.svc = LinearSVC()
        t = time.time()
        self.svc.fit(self.X_train, self.y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')

    def test_accuracy(self):
        print('Test Accuracy of SVC = ', round(self.svc.score(self.X_test, self.y_test), 4))
        t = time.time()
        print('prediction time for single image: ', t)
        n_predict = 10
        print('My SVC predicts: ', self.svc.predict(self.X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', self.y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

mod = model()
mod.get_features()
mod.train()
mod.test_accuracy()
svc = mod.svc
X_scaler = mod.X_scaler

data = {
    'svc': svc,
    'X_scaler': X_scaler,
    'orient': settings.orient,
    'pix_per_cell': settings.pix_per_cell,
    'cell_per_block': settings.cell_per_block,
    'spatial_size': settings.spatial_size,
    'hist_bins': settings.hist_bins
}

with open('svc_pickle.p', 'wb') as handle:
    pickle.dump(data, handle, protocol = pickle.HIGHEST_PROTOCOL)


