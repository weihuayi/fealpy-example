import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from mpi4py import MPI
from numpy.linalg import norm

from fealpy.parallel import CSRMatrixCommToplogy, NumCompComponent

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

print("size: ", size)
print("rank: ", rank)
print("name: ", name)


#A = np.array([[4, 0, 2], [0, 1, 0], [2, 0, 5]], dtype=np.float)
#b = np.array([1, 1, 1], dtype=np.float)
#A = csr_matrix(A)

data = sio.loadmat('amg.mat')
A = data['A'].tocsr()
b = data['b'].reshape(-1)

N = A.shape[0]


# 建立通信结构
ctop = CSRMatrixCommToplogy(comm, N)
A = ctop.get_parallel_operator(A)
lidx = ctop.get_local_idx()
c = NumCompComponent(ctop)

# 准备初始数据
maxit = 1
p = b
r = p[lidx].copy()
x = np.zeros(r.shape, dtype=np.float)

k = 0
sbuf = np.zeros(2, dtype=np.float)
rbuf = np.zeros(2, dtype=np.float)

Ap = A@p
sbuf[0] = np.sum(r**2)
sbuf[1] = np.sum(p[lidx]*Ap)
comm.Allreduce(sbuf, rbuf)

while k < maxit:

    alpha = rbuf[0]/rbuf[1]

    x += alpha*p[lidx]
    r -= alpha*Ap

    sbuf[0] = np.sum(r**2)
    rbuf[1] = rbuf[0]
    comm.Allreduce(sbuf[0:1], rbuf[0:1])

    print("rank: ", rank, "res:", rbuf[0])

    if np.sqrt(rbuf[0]) < 1e-8:
        break

    beta = rbuf[0]/rbuf[1]
    p[lidx] = r + beta*p[lidx]

    c.communicating(p)

    Ap = A@p
    sbuf[1] = np.sum(p[lidx]*Ap)
    comm.Allreduce(sbuf[1:], rbuf[1:])
    k += 1








