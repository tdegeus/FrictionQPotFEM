import GooseFEM as fem
import GMatElastoPlasticQPot.Cartesian2d as mat
import numpy as np
import matplotlib.pyplot as plt

# ----
# mesh
# ----

N = 3 ** 2
h = np.pi
L = h * N

mesh = fem.Mesh.Quad4.FineLayer(N, N, h)

coor = mesh.coor()
conn = mesh.conn()
dofs = mesh.dofs()

nnode = mesh.nnode()
ndim = mesh.ndim()
nelem = mesh.nelem()
nne = mesh.nne()

plastic = mesh.elementsMiddleLayer()
elastic = np.setdiff1d(np.arange(nelem), plastic)

left = mesh.nodesLeftOpenEdge()
right = mesh.nodesRightOpenEdge()
dofs[right, 0] = dofs[left, 0]
dofs[right, 1] = dofs[left, 1]

top = mesh.nodesTopEdge()
bottom = mesh.nodesBottomEdge()
nfix = top.size
iip = np.empty((2 * ndim * nfix), dtype=np.int);
iip[0 * nfix: 1 * nfix] = dofs[bottom, 0]
iip[1 * nfix: 2 * nfix] = dofs[bottom, 1]
iip[2 * nfix: 3 * nfix] = dofs[top, 0]
iip[3 * nfix: 4 * nfix] = dofs[top, 1]

vector = fem.VectorPartitioned(conn, dofs, iip)

quad = fem.Element.Quad4.Quadrature(vector.AsElement(coor))
nip = quad.nip()

u = vector.AllocateNodevec(0.0)
v = vector.AllocateNodevec(0.0)
a = vector.AllocateNodevec(0.0)
v_n = vector.AllocateNodevec(0.0)
a_n = vector.AllocateNodevec(0.0)

ue = vector.AllocateElemvec(0.0)
fe = vector.AllocateElemvec(0.0)

fmaterial = vector.AllocateNodevec(0.0)
fdamp = vector.AllocateNodevec(0.0)
fint = vector.AllocateNodevec(0.0)
fext = vector.AllocateNodevec(0.0)
fres = vector.AllocateNodevec(0.0)

Eps = quad.AllocateQtensor(2, 0.0)
Sig = quad.AllocateQtensor(2, 0.0)

# --------
# material
# --------

# parameters
c = 1.0
G = 1.0
K = 10.0 * G
rho = G / (c ** 2.0)
qL = 2.0 * np.pi / L
qh = 2.0 * np.pi / h
alpha = np.sqrt(2.0) * qL * c * rho
dt = 1.0 / (c * qh) / 10.0

# material definition
material = mat.Array2d([nelem, nip])

# assign elastic elements
I = np.zeros([nelem, nip], dtype=np.int)
I[elastic, :] = 1
material.setElastic(I, K, G)

# assign plastic elements
k = 2.0
epsy = 1e-5 + 1e-3 * np.random.weibull(k, [N, 1000])
epsy[:, 0] = 1e-5 + 1e-3 * np.random.random(N)
epsy = np.cumsum(epsy, 1)

I = np.zeros([nelem, nip], dtype=np.int)
idx = np.zeros([nelem, nip], dtype=np.int)
I[plastic, :] = 1
idx[plastic, :] = np.arange(N).reshape([-1, 1])
unit = np.ones(N)
material.setCusp(I, idx, K * unit, G * unit, epsy)

# initialize
material.check()
material.setStrain(Eps)

# -----------
# mass matrix
# -----------

x = vector.AsElement(coor)
nodalQuad = fem.Element.Quad4.Quadrature(x, fem.Element.Quad4.Nodal.xi(), fem.Element.Quad4.Nodal.w())
val_quad = rho * np.ones([nelem, nodalQuad.nip()])
m_M = fem.MatrixDiagonalPartitioned(conn, dofs, iip)
m_M.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad))

# --------------
# damping matrix
# --------------

x = vector.AsElement(coor)
nodalQuad = fem.Element.Quad4.Quadrature(x, fem.Element.Quad4.Nodal.xi(), fem.Element.Quad4.Nodal.w())
val_quad = alpha * np.ones([nelem, nodalQuad.nip()])
m_D = fem.MatrixDiagonal(conn, dofs)
m_D.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad))
