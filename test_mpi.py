import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

print("size: ", size)
print("rank: ", rank)
print("name: ", name)


data = sio.loadmat('test.mat')
A = data['A'].tocsr()
print(type(A))
N = A.shape[0]
x = np.ones(N, dtype=np.float)

#A = np.array([[4, 0, 2], [0, 1, 0], [2, 0, 5]], dtype=np.float)
#x = np.array([1, 1, 1], dtype=np.float)
#A = csr_matrix(A)

NN = N//size
R = N%size

count = NN*np.ones(size, dtype='i')
count[0:R] += 1
location = np.zeros(size+1, dtype='i')
location[1:] = np.cumsum(count)

A = A[location[rank]:location[rank+1], :]



t0 = MPI.Wtime()
x[location[rank]:location[rank+1]] = A@x
t1 = MPI.Wtime()
print("rank: ", rank, "compute time:", t1 - t0)
b = x[location[rank]:location[rank+1]]
comm.Allgatherv(b, [x, count, location[0:-1], MPI.DOUBLE])
t2 = MPI.Wtime()
print("rank: ", rank, "message time: ", t2 - t1)


