#!/usr/bin/env python3
#
import numpy as np
from fealpy.pde.poisson_model_2d import ffData
from fealpy.mesh import Quadtree
from fealpy.vem import PoissonVEMModel
from fealpy.tools.show import showmultirate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f1(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.exp(5*(x**2 + y**2))
    return val


def f2(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.exp(5*(x**2 + (y-1)**2))
    return val


theta = 0.2
beta = 0.5
pde = ffData()
qtree = pde.init_mesh(n=4, meshtype="quadtree")
mesh = qtree.to_pmesh()



for i in range(1):
    vem = PoissonVEMModel(pde, mesh, p=1, q=4)
    uI = vem.space.interpolation(f1)

    S = vem.project_to_smspace(uI)
    barycenter = vem.space.smspace.barycenter
    grad = S.grad_value(barycenter)
    eta = np.sqrt(np.sum(grad**2, axis=-1)*vem.area)

    isMarkedCell = qtree.refine_marker(eta, theta)
    qtree.refine(isMarkedCell)
    mesh = qtree.to_pmesh()


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)

for i in range(2):
    vem = PoissonVEMModel(pde, mesh, p=1, q=4)
    uI = vem.space.interpolation(f2)
    eta = vem.recover_estimate(uh=uI, residual=False)
    isMarkedCell = qtree.coarsen_marker(eta, beta)
    qtree.coarsen(isMarkedCell)
    mesh = qtree.to_pmesh()

vem = PoissonVEMModel(pde, mesh, p=1, q=4)
uI = vem.space.interpolation(f1)
fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
tri = qtree.leaf_cell(celltype='tri')
axes.plot_trisurf(x, y, tri, uI[:len(x)], cmap=plt.cm.jet, lw=0.0)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
