import GooseFEM as fem
import GMatElastoPlasticQPot.Cartesian2d as mat
import numpy as np
import GooseMPL as gplt
import matplotlib.pyplot as plt

plt.style.use(['goose', 'goose-latex'])

# ----
# mesh
# ----

N = 10
h = 0.1
L = h * N

mesh = fem.Mesh.Quad4.Regular(N, N, h)

coor = mesh.coor()
conn = mesh.conn()

virtual = np.arange(2) + coor.shape[0]

coor = np.vstack((coor, np.array([
    [L, L], # position irrelevant
    [L + h, L]]))) # position irrelevant

dofs = np.arange(coor.size).reshape(coor.shape)
disp = np.zeros(coor.shape)

nnode = coor.shape[0]
ndim = mesh.ndim()
nelem = mesh.nelem()
nne = mesh.nne()

left = mesh.nodesLeftOpenEdge()
right = mesh.nodesRightOpenEdge()
dofs[right, 0] = dofs[left, 0]
dofs[right, 1] = dofs[left, 1]

top = mesh.nodesTopEdge()
dofs[top, 0] = dofs[virtual[0], 0]
dofs[top, 1] = dofs[virtual[0], 1]

dofs = fem.Mesh.renumber(dofs)

bottom = mesh.nodesBottomEdge()
iip = np.hstack((
    dofs[bottom, 0],
    dofs[bottom, 1],
    dofs[virtual[1], :],
))

vector = fem.VectorPartitioned(conn, dofs, iip)

K = fem.MatrixPartitioned(conn, dofs, iip)
Solver = fem.MatrixPartitionedSolver()

quad = fem.Element.Quad4.Quadrature(vector.AsElement(coor))
nip = quad.nip()

# --------
# material
# --------

kappa = 1.0
mu = 1.0

material = mat.Array2d([nelem, nip])
material.setElastic(np.ones(material.shape()), kappa, mu)
material.check()

k = 0.05

C = material.Tangent()
K.assemble(quad.Int_gradN_dot_tensor4_dot_gradNT_dV(C))

C = np.array([
    [ k,  0, -k,  0],
    [ 0,  0,  0,  0],
    [-k,  0,  k,  0],
    [ 0,  0,  0,  0],
])t
K.add(dofs[virtual, :].ravel(), dofs[virtual, :].ravel(), C)

# -----
# solve
# -----

fig, axes = gplt.subplots(ncols=2)

disp = np.zeros(disp.shape)
u = []
f = []

for inc in range(2):

    for iiter in range(3):

        Eps = quad.SymGradN_vector(vector.AsElement(disp));
        material.setStrain(Eps)
        Sig = material.Stress()
        fmaterial = vector.AssembleNode(quad.Int_gradN_dot_tensor2_dV(Sig))

        print(fmaterial[virtual[0], :])

        disp[virtual[1], 0] = inc * 1.0

        fspring = np.zeros(coor.shape)
        fspring[virtual[0], :] = +k * (disp[virtual[1], :] - disp[virtual[0], :])
        fspring[virtual[1], :] = -k * (disp[virtual[1], :] - disp[virtual[0], :])

        # print(disp[virtual[0], 0], disp[virtual[1], 0], fspring[virtual[0], :], fspring[virtual[1], :])

        disp = Solver.Solve(K, fmaterial + fspring, disp)

        # Eps = quad.SymGradN_vector(vector.AsElement(disp));
        # material.setStrain(Eps)
        # Sig = material.Stress()
        # fmaterial = vector.AssembleNode(quad.Int_gradN_dot_tensor2_dV(Sig))

        # fspring = np.zeros(coor.shape)
        # fspring[virtual[0], :] = +k * (disp[virtual[1], :] - disp[virtual[0], :])
        # fspring[virtual[1], :] = -k * (disp[virtual[1], :] - disp[virtual[0], :])

        # print(fmaterial)





    u += [disp[virtual[1], 0]]
    f += [k * (disp[virtual[1], 0] - disp[virtual[0], 0])]

axes[0].plot(u, f)

gplt.patch(coor=coor + disp, conn=conn, axis=axes[1])
gplt.patch(coor=coor, conn=conn, linestyle='--', axis=axes[1])
plt.show()
