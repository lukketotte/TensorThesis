import matplotlib.image as mpimg
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

img = mpimg.imread('image\\lost_in_trans.jpg')
print(img.shape)
plt.imshow(img[:,:,0], plt.cm.Reds_r)
plt.show()

plt.imshow(img[:,:,1], plt.cm.Greens_r)
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
plt.show()

plt.imshow(img[:,:,2], plt.cm.Blues_r)
plt.axes().get_xaxis().set_ticks([])
plt.axes().get_yaxis().set_ticks([])
plt.show()