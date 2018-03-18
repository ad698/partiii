import numpy as np
import matplotlib.pyplot as plt
import sys

data = np.genfromtxt(sys.argv[1],skip_header=1)
header = np.genfromtxt(sys.argv[1],skip_footer=len(data))
density = data[:len(data)/2]
potential = data[len(data)/2:]
# plt.plot(potential[-90],'.')
plt.plot(np.log(density[29]),'.')
plt.show()
