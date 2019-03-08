import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from mpi4py import MPI

from fealpy.parallel import CSRMatrixCommToplogy, NumCompComponent

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

print("size: ", size)
print("rank: ", rank)
print("name: ", name)



#A = np.array([[4, 0, 2], [0, 1, 0], [2, 0, 5]], dtype=np.float)
#N = A.shape[0]
#p = np.array([1, 1, 1], dtype=np.float)
#A = csr_matrix(A)

data = sio.loadmat('amg.mat')
A = data['A'].tocsr()
p = data['b'].reshape(-1)

N = A.shape[0]

ctop = CSRMatrixCommToplogy(comm, N)

A = ctop.get_parallel_operator(A)
lidx = ctop.get_local_idx()

c = NumCompComponent(ctop)

for i in range(1):
    t0 = MPI.Wtime()

    p[lidx] = A@p

    t1 = MPI.Wtime()
    print("rank: ", rank, "compute time:", t1 - t0, "p:", p)

    c.communicating(p)

    t2 = MPI.Wtime()
    print("rank: ", rank, "message time: ", t2 - t1, "p:", p)




