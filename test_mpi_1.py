import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix
from mpi4py import MPI


class CommComponent():
    """
    通信构件， 负责建立通信数据结构和通信操作
    """
    def __init__(self, comm, N):
        size = comm.Get_size()
        NN = N//size
        RE = N%size

        # 任务划分
        self.comm = comm
        count = NN*np.ones(size, dtype='i')
        count[0:RE] += 1
        self.location = np.zeros(size+1, dtype='i')
        self.location[1:] = np.cumsum(count)

        # 数据发送接收数据结构
        self.neighbor = None
        self.sendds = {}
        self.recvds = {}

    def get_parallel_operator(self, A):
        rank = self.comm.Get_rank()
        A = A[self.location[rank]:self.location[rank+1]]
        self.build_comm_toplogy(A.indices)
        return A

    def get_parallel_array(self, a):
        rank = self.comm.Get_rank()
        lidx = np.arange(self.location[rank], self.location[rank+1])
        return ParaArray(lidx, array=a)

    def build_comm_toplogy(self, indices):
        size = comm.Get_size()
        rank = comm.Get_rank()

        indices = np.unique(indices)

        isNotLocal = (indices < self.location[rank]) | (indices >= self.location[rank+1])
        indices = indices[isNotLocal]

        N = self.location[-1]
        NN = N//size
        RE = N%size
        ranks = (indices - RE)//NN
        ranks[ranks<0] = 0
        self.neighbor = set(ranks)

        for r in self.neighbor:
            self.recvds[r] = indices[ranks==r]
            data = np.array(len(self.recvds[r]), dtype='i')
            comm.Isend(data, dest=r, tag=rank)

        for r in self.neighbor:
            data = np.zeros(1, dtype='i')
            req = comm.Irecv(data, source=r, tag=r)
            req.Wait()
            self.sendds[r] = np.zeros(data[0], dtype='i') 

        for r in self.neighbor: 
            data = self.recvds[r]
            comm.Isend(data, dest=r, tag=rank) 

        for r in self.neighbor:  
            data = self.sendds[r]
            req = comm.Irecv(data, source=r, tag=r)
            req.Wait()
        
    def communicating(self, parray):
        for r in self.neighbor: 
            data = parray[self.sendds[r]]
            comm.Isend(data, dest=r, tag=rank) 
        for r in self.neighbor:  
            data = np.zeros(len(self.recvds[r]), dtype=parray.dtype)
            req = comm.Irecv(data, source=r, tag=r)
            req.Wait()
            parray[self.recvds[r]] = data


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


## 准备数据
#A = np.array([[4, 0, 2], [0, 1, 0], [2, 0, 5]], dtype=np.float)
#print(A)
#N = A.shape[0]
#x = np.array([1, 2, 3], dtype=np.float)
#A = csr_matrix(A)

cc = CommComponent(comm, N)
pA = cc.get_parallel_operator(A)
px = cc.get_parallel_array(x)


for i in range(1):
    # 矩阵乘向量运算
    t0 = MPI.Wtime()
    px.update(pA@px)
    t1 = MPI.Wtime()
    print("rank: ", rank, "compute time:", t1 - t0)

    # 通信
    cc.communicating(px)
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



