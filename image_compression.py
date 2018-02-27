import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
# from scipy.misc import face, imresize
from tensorly.decomposition import tucker
import matplotlib.image as mpimg
from math import ceil
import time

from utils_np import *
from parafac_np import parafac

from core_parafac_analysis import *

### DONT RUN THIS ON THE LAPTOP, ITS RIP

# img=mpimg.imread('stinkbug.png')
img = mpimg.imread('image\\lost_in_trans.jpg')

# plt.imshow(img)
# plt.show()
img = np.array(img, dtype = np.float64)

# Get tucker core
img_tl = tl.tensor(img)
tucker_rank = [30, 80, 3]
core, tucker_factors = tucker(img_tl, ranks=tucker_rank, 
	init='random', tol=10e-5, random_state=1234, n_iter_max = 100,
	verbose = True)

core_np = tl.to_numpy(core)

# train on original data and core 
max_r = 40
t0 = time.time()
original_data_error = error_parafac(tensor = img, max_rank = max_r, 
	init = "random", verbose = True)
t1 = time.time()
print(t1 - t0)
t0 = time.time()
core_data_error = error_parafac(tensor = core_np, max_rank = max_r,
	init = "random", verbose = False)
t1 = time.time()
print(t1 - t0)
xint = range(0, max_r + 1, 5)

plt.plot(original_data_error)
plt.plot(core_data_error)
plt.ylabel("Training Error")
plt.xlabel("$Rank_{CP}$")
plt.title("Image, core 50% compression")
plt.legend(["Original data", "Core tensor"], loc = "upper right")
plt.grid(True)
plt.xticks(xint)
plt.show()




"""
def to_image(tensor, tensorly = True):
    # Convert from a float dtype back to uint8 for
    # img plotting
    if tensorly:
    	im = tl.to_numpy(tensor)
    else:
    	im = tensor
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# colour img is a pixel x pixel x colour tensor
tucker_rank = [30, 80, 3]

img = tl.tensor(img)

core, tucker_factors = tucker(img, ranks=tucker_rank, 
	init='random', tol=10e-5, random_state=1234, n_iter_max = 100,
	verbose = True)

tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)
print("reconstructed")

plt.figure()
plt.imshow(to_image(tucker_reconstruction))
plt.title("Tucker, 50 % compression")
plt.show()

# lets try parafac
pc = parafac(init = "random")

img = mpimg.imread('image\\lost_in_trans.jpg')
img = np.array(img, dtype = np.float64)

pc.X_data = img
r = 40
pc.rank = r
pc.init_factors()
pc.parafac_als()
X_hat = pc.reconstruct_X()
X_hat = to_image(X_hat, tensorly = False)

plt.figure
plt.imshow(X_hat)
plt.title("PARAFAC, rank %d" % r)
plt.show()

img = mpimg.imread('image\\lost_in_trans.jpg')

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ax.set_axis_off()
ax.imshow(img)
ax.set_title('original')

ax = fig.add_subplot(2, 2, 2)
ax.set_axis_off()
ax.imshow(to_image(tucker_reconstruction))
ax.set_title('Tucker, 50% compression')

ax = fig.add_subplot(2, 2, 3)
ax.set_axis_off()
ax.imshow(X_hat)
ax.set_title('Parafac, rank = 40')

# plt.tight_layout()
plt.show()
"""