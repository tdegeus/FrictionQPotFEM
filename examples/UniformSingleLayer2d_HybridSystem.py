import matplotlib.pyplot as plt
import numpy as np

ret = np.genfromtxt('UniformSingleLayer2d_HybridSystem.txt', delimiter=",")

fig, ax = plt.subplots()
ax.plot(ret[:, 0], ret[:, 1])
plt.show()
