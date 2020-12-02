import GooseFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np

# mesh
# ----

# define mesh
mesh = GooseFEM.Mesh.Quad4.FineLayer(13, 13)

# mesh dimensions
nelem = mesh.nelem()
nne = mesh.nne()
ndim = mesh.ndim()

# mesh definitions
coor = mesh.coor()
conn = mesh.conn()
dofs = mesh.dofs()

# node sets
nodesLft = mesh.nodesLeftOpenEdge()
nodesRgt = mesh.nodesRightOpenEdge()
nodesTop = mesh.nodesTopEdge()
nodesBot = mesh.nodesBottomEdge()

# element sets
plastic = mesh.elementsMiddleLayer()
elastic = np.setdiff1d(np.arange(nelem), plastic)

# periodicity and fixed displacements DOFs
# ----------------------------------------

dofs[nodesRgt, :] = dofs[nodesLft, :]

dofs = GooseFEM.Mesh.renumber(dofs)

iip = np.concatenate((
    dofs[nodesBot, 0],
    dofs[nodesBot, 1],
    dofs[nodesTop, 0],
    dofs[nodesTop, 1]
))

# simulation variables
# --------------------

# vector definition
vector = GooseFEM.VectorPartitioned(conn, dofs, iip)

# allocate system matrix
K = GooseFEM.MatrixPartitioned(conn, dofs, iip)
Solver = GooseFEM.MatrixPartitionedSolver()

# nodal quantities
disp = np.zeros(coor.shape)
fint = np.zeros(coor.shape)
fext = np.zeros(coor.shape)
fres = np.zeros(coor.shape)

# element vectors
ue = np.empty((nelem, nne, ndim))
fe = np.empty((nelem, nne, ndim))
Ke = np.empty((nelem, nne * ndim, nne * ndim))

# element/material definition
# ---------------------------

# element definition
quad = GooseFEM.Element.Quad4.Quadrature(vector.AsElement(coor))
nip = quad.nip()

# material definition
mat = GMat.Array2d([nelem, nip])

I = np.zeros(mat.shape(), dtype='int')
I[elastic, :] = 1
mat.setElastic(I, 10.0, 1.0)

I = np.zeros(mat.shape(), dtype='int')
I[plastic, :] = 1
mat.setElastic(I, 10.0, 1.0)

# solve
# -----

# strain
ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)

# pre-stress
trigger = plastic[int(len(plastic) / 2.0)]
Sigstar = quad.AllocateQtensor(2, 0.0)
Sigstar[trigger, :, 0, 0] = +1.0
Sigstar[trigger, :, 1, 1] = -1.0

# stress & tangent
mat.setStrain(Eps)
Sig = mat.Stress() + Sigstar
C = mat.Tangent()

# internal force
fe = quad.Int_gradN_dot_tensor2_dV(Sig)
fint = vector.AssembleNode(fe)

# stiffness matrix
Ke = quad.Int_gradN_dot_tensor4_dot_gradNT_dV(C)
K.assemble(Ke)

# residual
fres = fext - fint

# solve
disp = Solver.Solve(K, fres, disp)

# post-process
# ------------

# compute strain and stress
ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
Sig = mat.Stress() + Sigstar

print('Sig = ')
print(GMat.Deviatoric(Sig[trigger]))
print('Eps = ')
print(GMat.Deviatoric(Eps[trigger]))

# internal force
fe = quad.Int_gradN_dot_tensor2_dV(Sig)
fint = vector.AssembleNode(fe)

# apply reaction force
fext = vector.Copy_p(fint, fext)

# average stress per node
dV = quad.AsTensor(2, quad.dV())
Sig = np.average(Sig, weights=dV, axis=1)

# plot
# ----

import matplotlib.pyplot as plt
import GooseMPL as gplt

plt.style.use(['goose', 'goose-latex'])

sig = GMat.Sigd(Sig)

fig, ax = plt.subplots()
gplt.patch(coor=coor + disp, conn=conn, cindex=sig, cmap='Reds', axis=ax, clim=(0, 0.5))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)
fig.savefig('perturbation_pure-shear.pdf')
plt.close(fig)
