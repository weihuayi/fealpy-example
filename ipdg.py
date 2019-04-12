import sys
import numpy as np  
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData as PDE 
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.quadrature import GaussLegendreQuadrature

p = 1
q = 3
n = 1
pde = PDE()

mesh = pde.init_mesh(0)
integrator = mesh.integrator(q)
space = LagrangeFiniteElementSpace(mesh, p, spacetype='D') 
area = mesh.entity_measure('cell')
A = space.stiff_matrix(integrator, area)
F = space.source_vector(pde.source, integrator, area)

edge = mesh.entity('edge')
edge2cell = mesh.ds.edge_to_cell()
n = mesh.edge_unit_normal()
h = mesh.entity_measure('edge')

qf = GaussLegendreQuadrature(2)
bcs, ws = qf.quadpts, qf.weights

<<<<<<< HEAD
phi = space.edge_basis(bcs, edge2cell[:, 0], edge2cell[:, 2])
gphi = space.edge_grad_basis(bcs, edge2cell[:, 0], edge2cell[:, 2])
=======
NE = mesh.number_of_edges()
NQ = len(bcs)
ldof = space.number_of_local_dofs()
shape = (NE, NQ, 2*ldof)

phi = np.zeros(shape)
ngphi = np.zeros(shape)

phi[..., 0:ldof] = space.edge_basis(bcs, edge2cell[:, 0], edge2cell[:, 2])
gphi0 = space.edge_grad_basis(bcs, edge2cell[:, 0], edge2cell[:, 2])
ngphi[..., 0:ldof] = np.einsum('ijkm, im->ijk', gphi0, n) 

phi[..., ldof:] = -space.edge_basis(bcs, edge2cell[:, 1] , edge2cell[:, 3])
gphi1 = space.edge_grad_basis(bcs, edge2cell[:, 1], edge2cell[:, 3])
ngphi[..., ldof:] = np.einsum('ijkm, im->ijk', gphi1, n)

E = np.einsum('ijk, ijm, i->ikm', phi, ngphi, h)
print(E)
>>>>>>> 44858577285a3df937486260bbe8b30d33882709
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

