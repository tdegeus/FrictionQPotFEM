import numpy as np
import GooseFEM as gf
import FrictionQPotFEM.Generic2d as model

layer_elas = gf.Mesh.Quad4.Regular(20, 6)
layer_plas = gf.Mesh.Quad4.Regular(20, 1)

stich = gf.Mesh.Stitch()

x0 = layer_elas.coor()
x1 = layer_plas.coor()
x2 = layer_elas.coor()

x1[:, 1] += np.max(x0[:, 1])
x2[:, 1] += np.max(x1[:, 1])

stich.push_back(x0, layer_elas.conn())
stich.push_back(x1, layer_plas.conn())
stich.push_back(x2, layer_elas.conn())

drive0 = stich.nodeset(np.setdiff1d(np.arange(layer_elas.nnode()), layer_elas.nodesRightEdge()), 0)
drive1 = stich.nodeset(np.setdiff1d(np.arange(layer_elas.nnode()), layer_elas.nodesRightEdge()), 2)

left = stich.nodeset([
    layer_elas.nodesLeftOpenEdge(),
    layer_plas.nodesLeftEdge(),
    layer_elas.nodesLeftOpenEdge(),
])

right = stich.nodeset([
    layer_elas.nodesRightOpenEdge(),
    layer_plas.nodesRightEdge(),
    layer_elas.nodesRightOpenEdge(),
])

bottom = stich.nodeset(layer_elas.nodesBottomEdge(), 0)
top = stich.nodeset(layer_elas.nodesTopEdge(), 2)

nelem = stich.nelem()
coor = stich.coor()
conn = stich.conn()
dofs = stich.dofs()
dofs[right, :] = dofs[left, :]
dofs = gf.Mesh.renumber(dofs)
iip = np.concatenate((dofs[bottom, :].ravel(), dofs[top, :].ravel()))

elas = list(stich.elemmap(0)) + \
       list(stich.elemmap(2))

plas = list(stich.elemmap(1))

system = model.HybridSystem(coor, conn, dofs, iip, elas, plas)
system.setMassMatrix(1.0 * np.ones((nelem)))
system.setDampingMatrix(0.01 * np.ones((nelem)))
system.setElastic(1.0 * np.ones((len(elas))), 1.0 * np.ones((len(elas))))
system.setPlastic(1.0 * np.ones((len(plas))), 1.0 * np.ones((len(plas))), np.cumsum(0.05 * np.ones((len(plas), 100)), axis=1))
system.setDt(0.1)

u = system.u()
u = system.u()
u[top, 0] += 0.2
system.setU(u)
print(system.minimise())

u = system.u()
u[top, 0] += 0.2
system.setU(u)
print(system.minimise())

u = system.u()
u[top, 0] += 0.2
system.setU(u)
print(system.minimise())

u = system.u()
u[top, 0] += 0.2
system.setU(u)
print(system.minimise())

u = system.u()
u[top, 0] += 0.2
system.setU(u)
print(system.minimise())

u = system.u()
u[top, 0] += 0.2
system.setU(u)
print(system.minimise())

u = system.u()
fext = system.fext()

# print(fext[drive3, 0])
# # print(fext[drive1[1]])
# # print(fext[drive2[1]])
# # print(fext[drive3[1]])
# # print(fext)
# # # # print(u)

















c = np.zeros((stich.nelem()))
for i in range(3):
    c[stich.elemmap(i)] = i

# import sys
# import os
# import re
# import subprocess
# import shutil
# import numpy as np
# import h5py
import matplotlib.pyplot as plt
import GooseMPL as gplt
# import cppcolormap as cm

plt.style.use(['goose', 'goose-latex'])

fig, ax = plt.subplots()
gplt.patch(coor=coor + u, conn=conn, cindex=c)
# ax.plot(coor[left, 0], coor[left, 1], c='r', marker='o', ls='none')
# ax.plot(coor[right, 0], coor[right, 1], c='g', marker='o', ls='none')
# ax.plot(coor[top, 0], coor[top, 1], c='b', marker='o', ls='none')
# ax.plot(coor[bottom, 0], coor[bottom, 1], c='y', marker='o', ls='none')
# ax.plot(coor[drive0, 0], coor[drive0, 1], c='b', marker='o', ls='none')
# ax.plot(coor[drive1, 0], coor[drive1, 1], c='c', marker='o', ls='none')
# ax.plot(coor[drive2, 0], coor[drive2, 1], c='m', marker='o', ls='none')
# ax.plot(coor[drive3, 0], coor[drive3, 1], c='y', marker='o', ls='none')
plt.show()

