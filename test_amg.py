#!/usr/bin/env python3
#

import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix

from fealpy.solver import AMGSolver


data = sio.loadmat('test.mat')
A = data['A'].tocsr()
b = data['b'].reshape(-1)

solver = AMGSolver()
solver.setup(A)
solver.coarsen_rs(theta=0.025)
