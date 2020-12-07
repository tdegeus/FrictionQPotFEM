import GooseFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np
import matplotlib as mpl
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

    ue = vector.AsElement(disp)
    Eps = quad.SymGradN_vector(ue)
    mat.setStrain(Eps)

    sigeq = GMat.Sigd(np.average(mat.Stress(), weights=quad.AsTensor(2, quad.dV()), axis=1))
    E = np.average(mat.Energy(), weights=quad.dV(), axis=1)

    return GMat.Deviatoric(Eps[trigger, 0]), trigger, disp, sigeq, E, coor, conn, vector, quad, mat

# Effect of perturbation
# ----------------------

Sigstar_s = np.array([
    [ 0.0, +1.0],
    [+1.0,  0.0]])

Sigstar_p = np.array([
    [+1.0,  0.0],
    [ 0.0, -1.0]])

Epsstar_s, trigger, u_s, sigeq_s, energy_s, coor, conn, vector, quad, mat = ComputePerturbation(Sigstar_s)
Epsstar_p, trigger, u_p, sigeq_p, energy_p, coor, conn, vector, quad, mat = ComputePerturbation(Sigstar_p)

# Plot - simple shear
# -------------------

fig, ax = plt.subplots()

gplt.patch(coor=coor + u_s, conn=conn, cindex=sigeq_s, cmap='Reds', axis=ax, clim=(0, 0.5))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='Reds', norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5))
sm.set_array([])

ticks = [0.0, +0.5]
labels = [0.0, +0.5]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels(['{0:.1f}'.format(i) for i in labels])

fig.savefig('perturbation_simple-shear.pdf')
plt.close(fig)

# Plot - pure shear
# -----------------

fig, ax = plt.subplots()

gplt.patch(coor=coor + u_p, conn=conn, cindex=sigeq_p, cmap='Reds', axis=ax, clim=(0, 0.5))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='Reds', norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5))
sm.set_array([])

ticks = [0.0, +0.5]
labels = [0.0, +0.5]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels(['{0:.1f}'.format(i) for i in labels])

fig.savefig('perturbation_pure-shear.pdf')
plt.close(fig)

# Plot - simple shear
# -------------------

fig, ax = plt.subplots()

gplt.patch(coor=coor + u_s, conn=conn, cindex=energy_s, cmap='RdGy_r', axis=ax, clim=(-0.1, 0.1))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='RdGy_r', norm=mpl.colors.Normalize(vmin=-0.1, vmax=+0.1))
sm.set_array([])

ticks = [-0.1, 0.0, +0.1]
labels = [-0.1, 0.0, +0.1]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels(['{0:.1f}'.format(i) for i in labels])

fig.savefig('perturbation_simple-shear_energy.pdf')
plt.close(fig)

# Plot - pure shear
# -----------------

fig, ax = plt.subplots()

gplt.patch(coor=coor + u_p, conn=conn, cindex=energy_p, cmap='RdGy_r', axis=ax, clim=(-0.1, 0.1))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='RdGy_r', norm=mpl.colors.Normalize(vmin=-0.1, vmax=+0.1))
sm.set_array([])

ticks = [-0.1, 0.0, +0.1]
labels = [-0.1, 0.0, +0.1]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels(['{0:.1f}'.format(i) for i in labels])

fig.savefig('perturbation_pure-shear_energy.pdf')
plt.close(fig)

# Plot - simple shear
# -------------------

fig, ax = plt.subplots()

gplt.patch(coor=coor - u_s, conn=conn, cindex=sigeq_s, cmap='Reds', axis=ax, clim=(0, 0.5))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='Reds', norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5))
sm.set_array([])

ticks = [0.0, +0.5]
labels = [0.0, +0.5]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels(['{0:.1f}'.format(i) for i in labels])

fig.savefig('perturbation_minus-simple-shear.pdf')
plt.close(fig)

# Plot - pure shear
# -----------------

fig, ax = plt.subplots()

gplt.patch(coor=coor - u_p, conn=conn, cindex=sigeq_p, cmap='Reds', axis=ax, clim=(0, 0.5))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='Reds', norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5))
sm.set_array([])

ticks = [0.0, +0.5]
labels = [0.0, +0.5]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels(['{0:.1f}'.format(i) for i in labels])

fig.savefig('perturbation_minus-pure-shear.pdf')
plt.close(fig)

# Plot - simple shear
# -------------------

fig, ax = plt.subplots()

gplt.patch(coor=coor - u_s, conn=conn, cindex=energy_s, cmap='RdGy_r', axis=ax, clim=(-0.1, 0.1))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='RdGy_r', norm=mpl.colors.Normalize(vmin=-0.1, vmax=+0.1))
sm.set_array([])

ticks = [-0.1, 0.0, +0.1]
labels = [-0.1, 0.0, +0.1]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels(['{0:.1f}'.format(i) for i in labels])

fig.savefig('perturbation_minus-simple-shear_energy.pdf')
plt.close(fig)

# Plot - pure shear
# -----------------

fig, ax = plt.subplots()

gplt.patch(coor=coor - u_p, conn=conn, cindex=energy_p, cmap='RdGy_r', axis=ax, clim=(-0.1, 0.1))
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=ax)

ax.axis('equal')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='RdGy_r', norm=mpl.colors.Normalize(vmin=-0.1, vmax=+0.1))
sm.set_array([])

ticks = [-0.1, 0.0, +0.1]
labels = [-0.1, 0.0, +0.1]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels(['{0:.1f}'.format(i) for i in labels])

fig.savefig('perturbation_minus-pure-shear_energy.pdf')
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

