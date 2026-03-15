import numpy as np
import matplotlib.pyplot as plt
import os

nodes = np.loadtxt(os.path.join(os.getcwd(), 'nodes_log.csv'), delimiter=',')

nodes = nodes.reshape(100, 100)

plt.imshow(nodes, origin='lower')
plt.xlabel('Node index')
plt.ylabel('Input index')
plt.colorbar()

plt.show()