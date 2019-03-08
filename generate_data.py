import numpy as np
import scipy.io as sio

from fealpy.fem.PoissonFEMModel import PoissonFEMModel
from fealpy.pde.poisson_2d import CosCosData as PDE 
from fealpy.boundarycondition import DirichletBC

pde = PDE()
mesh = pde.init_mesh(3)
print(mesh.number_of_nodes())
integrator = mesh.integrator(3)
fem = PoissonFEMModel(pde, mesh, 1, integrator)

bc = DirichletBC(fem.space, pde.dirichlet)

A = fem.get_left_matrix()
b = fem.get_right_vector()
AD, b = bc.apply(A, b)
data = {"A":AD, "b":b}
sio.savemat('amg.mat', data)


