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
N = A.shape[0]
x = np.ones(N, dtype=np.float)

#A = np.array([[4, 0, 2], [0, 1, 0], [2, 0, 5]], dtype=np.float)
#N = A.shape[0]
#x = np.array([1, 1, 1], dtype=np.float)
#A = csr_matrix(A)

NN = N//size
R = N%size

# 任务分解
count = NN*np.ones(size, dtype='i')
count[0:R] += 1
location = np.zeros(size+1, dtype='i')
location[1:] = np.cumsum(count)

# 每个进程只拿到连续的行
A = A[location[rank]:location[rank+1], :]


# 实际运算
t0 = MPI.Wtime()
x[location[rank]:location[rank+1]] = A@x
t1 = MPI.Wtime()
print("rank: ", rank, "compute time:", t1 - t0)


# 通信， 这里有改进的空间
b = x[location[rank]:location[rank+1]]
comm.Allgatherv(b, [x, count, location[0:-1], MPI.DOUBLE])
t2 = MPI.Wtime()
print("rank: ", rank, "message time: ", t2 - t1)


# 每个进程需要知道要发送何种信息到何进程 
# 
# 首先建立通信结构
# 1. 拿到所有的非零的列指标
# 2. 列指标 unique
# 3. 计算列指标所在的进程
# 4. 发送属于其它进程的指标到其它进程
# 5. 接收需要发送的指标信息
#
# 计算
# 1. 每个进程矩阵向量乘
# 2. 发送信息到其它进程 
# 
# 用到 点对点通信 
# 1. comm.Isend(data, dest=目标进程编号, tag=发送数据的进程编号) 
# 1. req=comm.Irecv(buffer, source=来源进程编号, tag=来进程编号)
#    req.Wait() # 这里要等待接收到数据
# 
# 并行的对象
# 1. 并行的网格
# 1. 并行的矩阵
# 1. 并行的向量
# 
# 通信构件
# 1. 建立通信数据结构
# 1. 通信



