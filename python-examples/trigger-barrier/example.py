import GooseFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np
import matplotlib.pyplot as plt
import GooseMPL as gplt

plt.style.use(['goose', 'goose-latex'])

np.random.seed(42)

def CheckEquilibrium(disp):

    # mesh
    # ----

    # define mesh
    mesh = GooseFEM.Mesh.Quad4.FineLayer(33, 33)

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

    # stress & tangent
    mat.setStrain(Eps)
    Sig = mat.Stress()
    C = mat.Tangent()

    # stiffness matrix
    Ke = quad.Int_gradN_dot_tensor4_dot_gradNT_dV(C)
    K.assemble(Ke)

    # residual force
    fe = quad.Int_gradN_dot_tensor2_dV(Sig)
    fint = vector.AssembleNode(fe)
    fext = vector.Copy_p(fint, fext)
    fres = fext - fint

    # solve
    u = Solver.Solve(K, fres, disp)

    assert np.isclose(np.linalg.norm(fres), 0)
    assert np.isclose(np.linalg.norm(disp - u), 0)


def ComputePerturbation(sigma_star_test):

    # mesh
    # ----

    # define mesh
    mesh = GooseFEM.Mesh.Quad4.FineLayer(33, 33)

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
    for q in range(nip):
        Sigstar[trigger, q, :, :] = sigma_star_test
        Sigstar[trigger, q, :, :] = sigma_star_test

    # stress & tangent
    mat.setStrain(Eps)
    Sig = mat.Stress() - Sigstar
    C = mat.Tangent()

    # stiffness matrix
    Ke = quad.Int_gradN_dot_tensor4_dot_gradNT_dV(C)
    K.assemble(Ke)

    # residual force
    fe = quad.Int_gradN_dot_tensor2_dV(Sig)
    fint = vector.AssembleNode(fe)
    fext = vector.Copy_p(fint, fext)
    fres = fext - fint

    # solve
    disp = Solver.Solve(K, fres, disp)

    # post-process
    # ------------

    # compute strain and stress
    ue = vector.AsElement(disp)
    Eps = quad.SymGradN_vector(ue)
    mat.setStrain(Eps)
    Sig = mat.Stress() - Sigstar

    # residual force
    fe = quad.Int_gradN_dot_tensor2_dV(Sig)
    fint = vector.AssembleNode(fe)
    fext = vector.Copy_p(fint, fext)
    fres = fext - fint

    # average stress per node
    dV = quad.AsTensor(2, quad.dV())
    sigeq = GMat.Sigd(np.average(Sig, weights=dV, axis=1))

    return GMat.Deviatoric(Sig[trigger, 0]), GMat.Deviatoric(Eps[trigger, 0]), trigger, disp

# ------------
# Perturbation
# ------------

Sigstar_s = np.array([
    [ 0.0, +1.0],
    [+1.0,  0.0]])

Sigstar_p = np.array([
    [+1.0,  0.0],
    [ 0.0, -1.0]])

Sigstar_s, Epsstar_s, trigger, u_s = ComputePerturbation(Sigstar_s)
Sigstar_p, Epsstar_p, trigger, u_p = ComputePerturbation(Sigstar_p)

fac_s = 0.2 / (Sigstar_s[0, 1] * Epsstar_s[0, 1])
fac_p = 0.2 / (Sigstar_p[0, 0] * Epsstar_p[0, 0])

if Epsstar_s[0, 1] < 0:
    fac_s *= -1
elif Epsstar_s[0, 1] > 0:
    fac_s = abs(fac_s)

if Epsstar_p[0, 0] < 0:
    fac_p *= -1
elif Epsstar_p[0, 0] > 0:
    fac_p = abs(fac_p)

Sigstar_s *= fac_s
Sigstar_p *= fac_p

Epsstar_s *= fac_s
Epsstar_p *= fac_p

u_s *= fac_s
u_p *= fac_p

# -------
# Example
# -------

# mesh
# ----

# define mesh
mesh = GooseFEM.Mesh.Quad4.FineLayer(33, 33)

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

# pre-stress
s_xx = 0.3 * np.random.random(len(plastic))
s_xy = 0.3 * np.random.random(len(plastic))
s_xx = np.roll(s_xx, 15)
s_xy = np.roll(s_xy, 15)

Sigstar = quad.AllocateQtensor(2, 0.0)

for q in range(nip):
    Sigstar[plastic, q, 0, 0] = +s_xx
    Sigstar[plastic, q, 1, 1] = -s_xx
    Sigstar[plastic, q, 0, 1] = +s_xy
    Sigstar[plastic, q, 1, 0] = +s_xy

# strain, stress & tangent
ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
Sig = mat.Stress() - Sigstar
C = mat.Tangent()

# stiffness matrix
Ke = quad.Int_gradN_dot_tensor4_dot_gradNT_dV(C)
K.assemble(Ke)

# residual force
fe = quad.Int_gradN_dot_tensor2_dV(Sig)
fint = vector.AssembleNode(fe)
fext = vector.Copy_p(fint, fext)
fres = fext - fint

# set fixed displacements
disp[nodesTop, 0] = +5.0

# solve
disp = Solver.Solve(K, fres, disp)

# Plot
# ----

dV = quad.AsTensor(2, quad.dV())
sigeq = GMat.Sigd(np.average(Sig, weights=dV, axis=1))

fig, ax = plt.subplots()
gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq, cmap='Reds', axis=ax, clim=(0, 1.0))
ax.axis('equal')
ax.set_axis_off()
fig.savefig('example_start.pdf')

# Trigger
# -------

ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
Sig = mat.Stress()

dgamma = Epsstar_s[0, 1]
dE = Epsstar_p[0, 0]

eps = float(GMat.Epsd(Eps[trigger, 0, :, :]))
Epsd = GMat.Deviatoric(Eps[trigger, 0, :, :])
gamma = Epsd[0, 1]
E = Epsd[0, 0]

epsy = 0.4
a = dE ** 2 + dgamma ** 2
b = 2 * (E * dE + gamma * dgamma)
c = eps ** 2 - epsy ** 2
eta = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

disp += (eta * u_s + eta * u_p)

ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
Sig = mat.Stress()

dV = quad.AsTensor(2, quad.dV())
sigeq = GMat.Sigd(np.average(Sig, weights=dV, axis=1))

fig, ax = plt.subplots()
gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq, cmap='Reds', axis=ax, clim=(0, 1.0))
ax.axis('equal')
ax.set_axis_off()
fig.savefig('example_end.pdf')

# plot trigger

disp = (eta * u_s + eta * u_p)

ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
Sig = mat.Stress()

dV = quad.AsTensor(2, quad.dV())
sigeq = GMat.Sigd(np.average(Sig, weights=dV, axis=1))

fig, ax = plt.subplots()
gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq, cmap='Reds', axis=ax, clim=(0, 1.0))
ax.axis('equal')
ax.set_axis_off()
fig.savefig('example_perturbation.pdf')
