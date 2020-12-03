import GooseFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np
import matplotlib.pyplot as plt
import GooseMPL as gplt

plt.style.use(['goose', 'goose-latex'])


def CheckEquilibrium(disp):

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


    return GMat.Deviatoric(Sig[trigger, 0]), GMat.Deviatoric(Eps[trigger, 0]), trigger, disp, coor, conn, sigeq, vector, quad, mat

# Effect of perturbation
# ----------------------

Sigstar_s = np.array([
    [ 0.0, +1.0],
    [+1.0,  0.0]])

Sigstar_p = np.array([
    [+1.0,  0.0],
    [ 0.0, -1.0]])

Sigstar_s, Epsstar_s, trigger, u_s, _, _, _, vector, quad, mat = ComputePerturbation(Sigstar_s)
Sigstar_p, Epsstar_p, trigger, u_p, _, _, _, vector, quad, mat = ComputePerturbation(Sigstar_p)

fac_s = 0.2 / (Sigstar_s[0, 1] * Epsstar_s[0, 1])
fac_p = 0.2 / (Sigstar_p[0, 0] * Epsstar_p[0, 0])

Sigstar_s *= fac_s
Sigstar_p *= fac_p

u_s *= fac_s
u_p *= fac_p

# Plot - simple shear
# -------------------

_, _, _, disp, coor, conn, sigeq, _, _, _ = ComputePerturbation(Sigstar_s)

fig, ax = plt.subplots()

gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq, cmap='Reds', axis=ax, clim=(0, 0.5))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

fig.savefig('perturbation_simple-shear.pdf')
plt.close(fig)

# Plot - pure shear
# -----------------

_, _, _, disp, coor, conn, sigeq, _, _, _ = ComputePerturbation(Sigstar_p)

fig, ax = plt.subplots()

gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq, cmap='Reds', axis=ax, clim=(0, 0.5))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

fig.savefig('perturbation_pure-shear.pdf')
plt.close(fig)

# Plot - combination
# ------------------

_, _, _, disp, coor, conn, sigeq, _, _, _ = ComputePerturbation(1.123 * Sigstar_s + 1.456 * Sigstar_p)

CheckEquilibrium(1.123 * u_s + 1.456 * u_p)

fig, ax = plt.subplots()

gplt.patch(coor=coor + disp, conn=conn, cindex=sigeq, cmap='Reds', axis=ax, clim=(0, 0.5))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

fig.savefig('perturbation_combination.pdf')
plt.close(fig)

# Phase diagram for strain and stress
# -----------------------------------

sig = np.zeros((101, 101))
eps = np.zeros((101, 101))

for i, p in enumerate(np.linspace(-1, 1, sig.shape[0])):
    for j, s in enumerate(np.linspace(-1, 1, sig.shape[1])):
        disp = s * u_s + p * u_p
        # CheckEquilibrium(disp)
        ue = vector.AsElement(disp)
        Eps = quad.SymGradN_vector(ue)
        mat.setStrain(Eps)
        Sig = mat.Stress()
        sig[i, j] = GMat.Sigd(Sig[trigger, 0])
        eps[i, j] = GMat.Epsd(Eps[trigger, 0])


fig, ax = plt.subplots()

h = ax.imshow(sig, cmap='jet', extent=[-1, +1, -1, +1])

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$p$')

fig.savefig('phase-diagram_sig.pdf')
plt.close(fig)


fig, ax = plt.subplots()

h = ax.imshow(eps, cmap='jet', extent=[-1, +1, -1, +1])

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$p$')

fig.savefig('phase-diagram_eps.pdf')
plt.close(fig)

# Phase diagram for energy
# ------------------------

energy = np.zeros((101, 101))
dV = quad.dV()

for i, p in enumerate(np.linspace(-1, 1, energy.shape[0])):
    for j, s in enumerate(np.linspace(-1, 1, energy.shape[1])):
        disp = s * u_s + p * u_p
        # CheckEquilibrium(disp)
        ue = vector.AsElement(disp)
        Eps = quad.SymGradN_vector(ue)
        mat.setStrain(Eps)
        energy[i, j] = np.average(mat.Energy(), weights=dV) * np.sum(dV)

fig, ax = plt.subplots()

h = ax.imshow(energy, cmap='jet', extent=[-1, +1, -1, +1])

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$p$')

fig.savefig('phase-diagram_energy.pdf')
plt.close(fig)

