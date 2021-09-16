import sys

import matplotlib.pyplot as plt
import numpy as np

ret = np.genfromtxt("Generic2d_System.txt", delimiter=",")

if len(sys.argv) == 2:
    test = np.genfromtxt(sys.argv[1], delimiter=",")
    assert np.allclose(ret, test)
    print("Check successful")

fig, ax = plt.subplots()
ax.plot(ret[:, 0], ret[:, 1])
plt.show()
