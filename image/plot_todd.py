import numpy as np
import matplotlib.pyplot as plt
import math
from math import ceil
import matplotlib.image as mpimg

img = mpimg.imread("todd.jpg")



fig = plt.figure()

ax = fig.add_subplot(2,2,1)
#ax.set_axis_off()
ax.imshow(img[:,:,0])
ax.set_title("Red channel")

ax = fig.add_subplot(2,2,2)
#ax.set_axis_off()
ax.imshow(img[:,:,1])
ax.set_title("Green channel")

ax = fig.add_subplot(2,2,3)
#ax.set_axis_off()
ax.imshow(img[:,:,2])
ax.set_title("Blue channel")
plt.tight_layout()
plt.show()




