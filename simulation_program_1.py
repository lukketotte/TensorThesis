from parafac_np import parafac
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import math

# X is 10x10x10 generated from N(2,8)
X = np.random.normal(2, 8, 10*10*10).reshape(10,10,10)
# X = np.random.exponential(5, 20*20*20).reshape(20,20,20)
# run tucker decomposition
Xt = tl.tensor(X)
core, factors = tucker(Xt , ranks = [5, 5,5])
tucker_reconstruction = tl.tucker_to_tensor(core, factors)
core = tl.to_numpy(core)

n_iter = 30
x_error_over_ranks = [None] * n_iter
g_error_over_ranks = [None] * n_iter

pc = parafac(init = "hosvd")
pc.X_data = X

pc_g = parafac(init = "hosvd")
pc_g.X_data = core

for rank in range(n_iter):
	# parafac on X
	pc.rank = rank
	pc.init_factors()
	error_x = pc.parafac_als()
	# parafac on Core
	pc_g.rank = rank
	pc_g.init_factors()
	error_g = pc_g.parafac_als()

	x_error_over_ranks[rank] = error_x[len(error_x) - 1]
	g_error_over_ranks[rank] = error_g[len(error_g) - 1]


xint = range(0, n_iter + 1, 2)


plt.plot(x_error_over_ranks)
plt.plot(g_error_over_ranks)
plt.ylabel('Training error')
plt.xlabel('$Rank_{CP}$')
plt.title("$N(\mu = 2, \sigma^2 = 8),\mathcal{X} \in \Re^{10x10x10}$")
plt.legend(['Original data', 'Core tensor(.5)'], loc='upper right')
plt.grid(True)
plt.xticks(xint)
plt.show()


"""
# TODO: how to add extras to subplots?
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot(x_error_over_ranks)
ax1.plot(g_error_over_ranks)

ax2 = fig.add_subplot(211)
ax2.plot(x_error_over_ranks)
ax2.plot(g_error_over_ranks)

plt.show()
"""