import matplotlib.image as mpimg
import glob
from find_cars import Car_Finder
import cv2

img = mpimg.imread('./test_images/test1.jpg')
cf = Car_Finder()

test_images = glob.glob("./test_images/*.jpg")
print('test_images:', test_images)
for image in test_images:
    img = mpimg.imread(image)
    out_image = cf.process_image(img)
    cv2.imwrite('./results/' + str(image) + 'processed.jpg', out_image)
