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


data = sio.loadmat('test.mat')
A = data['A'].tocsr()
b = data['b'].reshape(-1)
N = A.shape[0]




# 建立通信结构
ctop = CSRMatrixCommToplogy(comm, N)
A = ctop.get_parallel_operator(A)
c = NumCompComponent(ctop)

# 准备初始数据
maxit = 1
p = ctop.get_parallel_array(b)
r = p[p.lidx].copy()
x = np.zeros(r.shape, dtype=np.float)

k = 0
while k < maxit:
    Ap = A@p

    buf = np.zeros(2, dtype=np.float)
    buf[0] = np.sum(r**2)
    buf[1] = np.sum(p[p.lidx]*Ap)
    comm.Reduce(MPI.IN_PLACE, buf)
    print('rank: ', rank, 'buf: ', buf)





c.computing(A, px)




