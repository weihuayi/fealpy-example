import sys
import numpy as np  
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData as PDE 
from fealpy.functionspace import LagrangeFiniteElementSpace

p = 1
q = 3
n = 1
pde = PDE()

mesh = pde.init_mesh(n)
integrator = mesh.integrator(q)
space = LagrangeFiniteElementSpace(mesh, p, spacetype='D') 
area = mesh.entity_measure('cell')
A = space.stiff_matrix(integrator, area)
F = space.source_vector(pde.source, integrator, area)

edge = mesh.entity('edge')
edge2cell = mesh.ds.edge_to_cell()
n = mesh.edge_unit_normal()


print(F)
print(mesh.number_of_cells())
print(space.number_of_global_dofs())



fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

