import numpy as np
import scipy.io as sio

from fealpy.fem.PoissonFEMModel import PoissonFEMModel
from fealpy.pde.poisson_2d import CosCosData as PDE 
from fealpy.boundarycondition import DirichletBC

pde = PDE()
mesh = pde.init_mesh(4)
integrator = mesh.integrator(3)
fem = PoissonFEMModel(pde, mesh, 1, integrator)

bc = DirichletBC(fem.femspace, pde.dirichlet)

A = fem.get_left_matrix()
b = fem.get_right_vector()
AD, b = bc.apply(A, b)
data = {"A":AD, "b":b}
sio.savemat('test.mat', data)


