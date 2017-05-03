import matplotlib.image as mpimg

from find_cars import Car_Finder

img = mpimg.imread('./test_images/test1.jpg')
cf = Car_Finder()
out_img = cf.process_image(img)