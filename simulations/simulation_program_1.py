# sys.path.append('/TensorPython/')
from TensorPython import parafac_np
from parafac_np import parafac_np
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker

# X is 10x10x10 generated from N(2,8)
X = np.random.normal(2, 8, 10*10*10).reshape(10,10,10)

# run tucker decomposition
Xt = tl.tensor(X)
core, factors = tucker(Xt , ranks = [5,5,5])
tucker_reconstruction = tl.tucker_to_tensor(core, factors)
core = tl.to_numpy(core)

# parafac on X
pc = parafac(init = "hosvd")
pc.X_data = X
pc.rank = 2
pc.init_factors()
error_x = pc.parafac_als()
# parafac on Core
pc_g = parafac(init = "hosvd")
pc_g.X_data = core
pc_g.rank = 2
pc_g.init_factors()
error_g = pc_g.parafac_als()

print(error_x[len(error_x)])