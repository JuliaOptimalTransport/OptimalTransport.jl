import ot
import numpy as np

N = 1000
mu = np.random.rand(N, 1)
nu = np.random.rand(N, 1)
C = ot.utils.dist(mu, nu)
eps = 0.01

%timeit ot.sinkhorn(mu, nu, C, eps)



