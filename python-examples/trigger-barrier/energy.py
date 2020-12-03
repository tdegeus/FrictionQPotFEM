import GooseFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np
import matplotlib.pyplot as plt
import GooseMPL as gplt

plt.style.use(['goose', 'goose-latex'])

def ComputePerturbation(sigma_star):

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
    for q in range(nip):
        Sigstar[trigger, q, :, :] = sigma_star
        Sigstar[trigger, q, :, :] = sigma_star

    # stress & tangent
    mat.setStrain(Eps)
    Sig = mat.Stress() - Sigstar
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
    Sig = mat.Stress() - Sigstar

    # output
    # ------

    return GMat.Deviatoric(Sig[trigger, 0]), GMat.Deviatoric(Eps[trigger, 0]), disp

Sigstar_s = np.array([
    [ 0.0, +1.0],
    [+1.0,  0.0]])

Sigstar_p = np.array([
    [+1.0,  0.0],
    [ 0.0, -1.0]])

Sig_s, Eps_s, u_s = ComputePerturbation(Sigstar_s)
Sig_p, Eps_p, u_p = ComputePerturbation(Sigstar_p)

fac_s = 1.0 / (Sig_s[0, 1] * Eps_s[0, 1])
fac_p = 1.0 / (Sig_p[0, 0] * Eps_p[0, 0])


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

# Diagonalise system

ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
C = mat.Tangent()

Ke = quad.Int_gradN_dot_tensor4_dot_gradNT_dV(C)
K.assemble(Ke)

# Compute
# -------

Sigstar_s = np.array([
    [ 0.0, +1.0],
    [+1.0,  0.0]])

Sigstar_p = np.array([
    [+1.0,  0.0],
    [ 0.0, -1.0]])

trigger = plastic[int(len(plastic) / 2.0)]
Sigstar = quad.AllocateQtensor(2, 0.0)

dV = quad.dV()

energy = np.zeros((101, 101))

for i, p in enumerate(np.linspace(-1, 1, energy.shape[0])):
    for j, s in enumerate(np.linspace(-1, 1, energy.shape[1])):

        disp *= 0

        for q in range(nip):
            Sigstar[trigger, q, :, :] = s * Sigstar_s * fac_s + p * Sigstar_p * fac_p
            Sigstar[trigger, q, :, :] = s * Sigstar_s * fac_s + p * Sigstar_p * fac_p

        ue = vector.AsElement(disp)
        Eps = quad.SymGradN_vector(ue)
        mat.setStrain(Eps)
        Sig = mat.Stress() - Sigstar

        fe = quad.Int_gradN_dot_tensor2_dV(Sig)
        fint = vector.AssembleNode(fe)

        fres = fext - fint

        disp = Solver.Solve(K, fres, disp)

        ue = vector.AsElement(disp)
        Eps = quad.SymGradN_vector(ue)
        mat.setStrain(Eps)

        energy[i, j] = np.average(mat.Energy(), weights=dV) * np.sum(dV)

# Plot
# ----

fig, ax = plt.subplots()

h = ax.imshow(energy, cmap='jet', extent=[-1, +1, -1, +1])

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$p$')

fig.savefig('energy.pdf')
plt.close(fig)
