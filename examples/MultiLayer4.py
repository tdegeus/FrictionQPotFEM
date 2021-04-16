import numpy as np
import GooseFEM as gf
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model

layer_elas = gf.Mesh.Quad4.Regular(20, 6)
layer_plas = gf.Mesh.Quad4.Regular(20, 1)

stitch = gf.Mesh.Stitch()

x0 = layer_elas.coor()
x1 = layer_plas.coor()
x2 = layer_elas.coor()

x1[:, 1] += np.max(x0[:, 1])
x2[:, 1] += np.max(x1[:, 1])

stitch.push_back(x0, layer_elas.conn())
stitch.push_back(x1, layer_plas.conn())
stitch.push_back(x2, layer_elas.conn())

# drive0 = stitch.nodeset(np.setdiff1d(np.arange(layer_elas.nnode()), layer_elas.nodesRightEdge()), 0)
# drive1 = stitch.nodeset(np.setdiff1d(np.arange(layer_elas.nnode()), layer_elas.nodesRightEdge()), 2)

left = stitch.nodeset([
    layer_elas.nodesLeftOpenEdge(),
    layer_plas.nodesLeftEdge(),
    layer_elas.nodesLeftOpenEdge(),
])

right = stitch.nodeset([
    layer_elas.nodesRightOpenEdge(),
    layer_plas.nodesRightEdge(),
    layer_elas.nodesRightOpenEdge(),
])

bottom = stitch.nodeset(layer_elas.nodesBottomEdge(), 0)
top = stitch.nodeset(layer_elas.nodesTopEdge(), 2)

nelem = stitch.nelem()
coor = stitch.coor()
conn = stitch.conn()
dofs = stitch.dofs()
dofs[right, :] = dofs[left, :]
dofs[top[0], 0] = dofs[top[-1], 0]
dofs[bottom[0], 0] = dofs[bottom[-1], 0]
dofs = gf.Mesh.renumber(dofs)
iip = np.concatenate((dofs[bottom, :].ravel(), dofs[top, 1].ravel()))

elas = list(stitch.elemmap(0)) + \
       list(stitch.elemmap(2))

plas = list(stitch.elemmap(1))

system = model.System(coor, conn, dofs, iip, stitch.elemmap(), stitch.nodemap(), [False, True, False])
system.setMassMatrix(1.0 * np.ones((nelem)))
system.setDampingMatrix(0.01 * np.ones((nelem)))
system.setElastic(10.0 * np.ones((len(elas))), 1.0 * np.ones((len(elas))))
system.setPlastic(10.0 * np.ones((len(plas))), 1.0 * np.ones((len(plas))), np.cumsum(0.01 * np.ones((len(plas), 100)), axis=1))
system.setDt(0.1)
system.setDriveStiffness(1.0)

ubar = np.zeros((3, 2))
drive = np.zeros((3, 2), dtype=bool)
ubar[2, 0] = 0.5
drive[2, 0] = True
drive[0, 0] = True

u = system.u()
u[stitch.nodemap(2), 0] = 0.0
system.setU(u)

system.layerSetUbar(ubar, drive)

fdrive = system.fdrive()

# for i in range(50000):
#     system.timeStep()

# print(system.residual())
print(system.minimise())
print(system.layerUbar())
print(system.u())


# # vector = system.vector()
# # print(vector.AsDofs(fdrive))

# d = np.argwhere(np.sum(fdrive != 0, axis=1)).ravel()
# u = system.u()

# u[stitch.nodemap(2), 0] = drive * 0.8
# system.setU(u)

# vector = system.vector()
# quad = gf.Element.Quad4.Quadrature(vector.AsElement(coor))
# dV = quad.AsTensor(1, quad.dV())
# ue = vector.AsElement(u)
# ubar = quad.InterpQuad_vector(ue)




# # print(vector.AsElement(u))
# # print(ubar[stitch.elemmap(0), :, :])
# # print(ubar[stitch.elemmap(1), :, :])
# # print(ubar[stitch.elemmap(2), :, :])
# ubar0 = np.average(ubar[stitch.elemmap(0), :, :], weights=dV[stitch.elemmap(0), :, :], axis=(0, 1))
# ubar1 = np.average(ubar[stitch.elemmap(1), :, :], weights=dV[stitch.elemmap(1), :, :], axis=(0, 1))
# ubar2 = np.average(ubar[stitch.elemmap(2), :, :], weights=dV[stitch.elemmap(2), :, :], axis=(0, 1))

# fdrive = np.zeros(quad.shape_qtensor(1))
# fdrive[stitch.elemmap(2), :, 0] = ubar2[0] - drive
# fdrive = vector.AsNode(vector.AssembleDofs(quad.Int_N_vector_dV(fdrive)))
# print(fdrive)

# # # print(np.linalg.norm(system.fdrive()))
# # # print(system.minimise())
# # for i in range(10000):
# #     system.timeStep()

# # print(system.residual())
# # u = 100 * system.u()

# # print(system.fdrive())
# # print(system.fmaterial())

# # print(system.layerUbar(0))
# # print(system.layerUbar(1))
# # print(system.layerUbar(2))

c = np.zeros((stitch.nelem()))
for i in range(3):
    c[stitch.elemmap(i)] = i

import matplotlib.pyplot as plt
import GooseMPL as gplt

plt.style.use(['goose', 'goose-latex'])

x = coor + 10 * system.u()
fig, ax = plt.subplots()
gplt.patch(coor=x, conn=conn, cindex=c)
# # ax.plot(x[d, 0], x[d, 1], c='r', marker='o', ls='none')
# # # ax.plot(x[left, 0], x[left, 1], c='g', marker='o', ls='none')
# # # ax.plot(x[right, 0], x[right, 1], c='g', marker='o', ls='none')
# # # ax.plot(x[top, 0], x[top, 1], c='b', marker='o', ls='none')
# # # ax.plot(x[bottom, 0], x[bottom, 1], c='y', marker='o', ls='none')
# # # ax.plot(x[drive0, 0], x[drive0, 1], c='b', marker='o', ls='none')
# # # ax.plot(x[drive1, 0], x[drive1, 1], c='c', marker='o', ls='none')
# # # ax.plot(x[drive2, 0], x[drive2, 1], c='m', marker='o', ls='none')
# # # ax.plot(x[drive3, 0], x[drive3, 1], c='y', marker='o', ls='none')
plt.show()

