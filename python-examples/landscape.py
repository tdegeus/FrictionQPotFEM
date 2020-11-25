import numpy as np
import GooseFEM as fem
import GMatElastoPlasticQPot.Cartesian2d as mat
import matplotlib.pyplot as plt
import GooseMPL as gplt
import cppcolormap as cm

plt.style.use(['goose', 'goose-latex'])

fig, axes = gplt.subplots(ncols=2)

h = 1.0
mesh = fem.Mesh.Quad4.Regular(3, 3, h=h)
conn = mesh.conn()
coor = mesh.coor()
dofs = mesh.dofs()
nelem = mesh.nelem()

vector = fem.Vector(conn, dofs)
quad = fem.Element.Quad4.Quadrature(vector.AsElement(coor))
nip = quad.nip()
dV = quad.dV()

for sim in range(6):

    # disp = np.zeros(coor.shape)
    disp = 0.1 * np.random.random(coor.shape) - 0.05

    material = mat.Array2d([nelem, nip])

    plastic = np.array([4])
    elastic = np.setdiff1d(np.arange(nelem), plastic)
    N = len(plastic)

    mu = 1.0
    kappa = 10.0 * mu

    k = 2.0
    epsy = 1e-5 + 1e-3 * np.random.weibull(k, [N, 1000])
    epsy[:, 0] = 1e-5 + 1e-3 * np.random.random(N)
    epsy = np.cumsum(epsy, 1)

    I = np.zeros(material.shape())
    I[elastic, :] = 1
    material.setElastic(I, kappa, mu)

    I = np.zeros(material.shape())
    idx = np.zeros(material.shape())
    I[plastic, :] = 1
    idx[plastic, :] = 0
    material.setCusp(I, idx, kappa * np.ones(N), mu * np.ones(N), epsy)

    material.check()

    Gamma = np.linspace(0, 0.01, 100)
    Energy = []
    Energy_plastic = []

    for inc in range(len(Gamma)):
        print(inc)
        if inc > 0:
            disp[conn[plastic, :2], 0] -= 0.5 * (Gamma[inc] - Gamma[inc - 1]) * h
            disp[conn[plastic, 2:], 0] += 0.5 * (Gamma[inc] - Gamma[inc - 1]) * h
        Eps = quad.SymGradN_vector(vector.AsElement(disp))
        material.setStrain(Eps)
        E = material.Energy()
        Energy += [np.average(E, weights=dV)]
        Energy_plastic += [np.average(E[plastic, :], weights=dV[plastic, :])]

    Energy = np.array(Energy)
    Energy_plastic = np.array(Energy_plastic)

    axes[0].plot(Gamma, Energy - Energy[0], marker='.')
    axes[1].plot(Gamma, Energy_plastic - Energy_plastic[0], marker='.')

plt.show()

# fig, ax = plt.subplots()
# gplt.patch(coor=coor + disp, conn=conn, axis=ax)
# gplt.patch(coor=coor + disp0, conn=conn, linestyle='--', axis=ax)
# plt.show()



