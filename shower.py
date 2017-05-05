import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./results/heatmaps/551heatmap.jpg')
plt.imshow(image, cmap='hot')
plt.show()