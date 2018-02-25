import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize
from tensorly.decomposition import tucker
import matplotlib.image as mpimg
from math import ceil

from utils_np import *
from parafac_np import parafac

# img=mpimg.imread('stinkbug.png')
img = mpimg.imread('image\\space_odyssey.jpg')
print(type(img))
print(img.shape)
plt.imshow(img)
# plt.show()

"""
# hsov throws memory error, should catch it 
pc = parafac(init = "random", epochs = 100)
pc.X_data = img
pc.rank = 50
pc.init_factors()
pc.parafac_als(verbose = True)
xhat = pc.reconstruct_X()
"""
def to_image(tensor):
    # Convert from a float dtype back to uint8 for
    # img plotting
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# colour img is a pixel x pixel x colour tensor
tucker_rank = [432, 960, 3]

img = tl.tensor(img)

core, tucker_factors = tucker(img, ranks=tucker_rank, 
	init='random', tol=10e-5, random_state=1234, n_iter_max = 10)

tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_axis_off()
ax.imshow(img)
ax.set_title('original')

ax = fig.add_subplot(1, 2, 2)
ax.set_axis_off()
ax.imshow(to_image(tucker_reconstruction))
ax.set_title('Tucker')

plt.tight_layout()
plt.show()