#!/usr/bin/env python3
#
"""
The adaptive vem example for poisson problem
"""

import numpy as np
import sys

from fealpy.pde.poisson_model_2d import LShapeRSinData
from fealpy.vem import PoissonVEMModel
from fealpy.tools.show import showmultirate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

maxit = 10 

pde = LShapeRSinData()
qtree = pde.init_mesh(n=4, meshtype="quadtree")
theta = 0.5
beta = 0.5

k = maxit - 5 
errorType = ['$\| u_I - u_h \|_{l_2}$',
             '$\|\\nabla u_I - \\nabla u_h\|_A$',
             '$\| u - \Pi^\Delta u_h\|_0$',
             '$\|\\nabla u - \\nabla \Pi^\Delta u_h\|$',
             '$\|\\nabla \Pi^\Delta u_h - \Pi^\Delta G(\\nabla \Pi^\Delta u_h) \|$'
             ]
Ndof = np.zeros((maxit,), dtype=np.int)
errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
mesh = qtree.to_pmesh()

for i in range(maxit):
    print('step:', i)
    vem = PoissonVEMModel(pde, mesh, p=1, q=4)
    vem.solve()
    eta = vem.recover_estimate(residual=True)
#    print(
#            "std(eta):", np.std(eta)/np.mean(eta),
#            "max(eta):", np.max(eta),
#            "min(eta):", np.min(eta),
#            sep='\n'
#            )

    Ndof[i] = vem.space.number_of_global_dofs()
    print("Ndof[i]:", Ndof[i])
    errorMatrix[0, i] = vem.l2_error()
    errorMatrix[1, i] = vem.uIuh_error()
    errorMatrix[2, i] = vem.L2_error()
    errorMatrix[3, i] = vem.H1_semi_error()
    errorMatrix[0, i] = np.sqrt(np.sum(eta**2))
    if i < maxit - 1:
        data = {"uh": vem.uh.copy()}
        isMarkedCell = qtree.refine_marker(eta, theta)
        qtree.refine(isMarkedCell, data=data)
        mesh = qtree.to_pmesh()
        print("After refine:", mesh.number_of_nodes())
        vem.reinit(mesh, 1)
        vem.uh[:] = data["uh"]
        eta = vem.recover_estimate(residual=True)
        isMarkedCell = qtree.coarsen_marker(eta, beta)
        qtree.coarsen(isMarkedCell)
        mesh = qtree.to_pmesh()
        print("After coarsen:", mesh.number_of_nodes())
    

mesh.add_plot(plt, cellcolor='w')

fig2 = plt.figure()
fig2.set_facecolor('white')
axes = fig2.gca(projection='3d')
x = mesh.node[:, 0]
y = mesh.node[:, 1]
tri = qtree.leaf_cell(celltype='tri')
axes.plot_trisurf(x, y, tri, vem.uh[:len(x)], cmap=plt.cm.jet, lw=0.0)

showmultirate(plt, k, Ndof, errorMatrix, errorType)
plt.show()
