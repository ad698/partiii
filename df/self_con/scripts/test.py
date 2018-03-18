from actest_py import *
import numpy as np
G = GalPot("../../Torus/pot/DB97Mod1.Tpot")
O = Orbit(G, 1e-7)
A = Actions_AxisymmetricStackel_Fudge(G, -4.)
X = np.array([7.89732, 0.00001, 0.0352859, 53.428, 221.1465, 127.5689])
print G(X[:3])
T = IterativeTorusMachine(A, G, 1e-8, 5, 1e-3)
print A.actions(X)
print T.actions_and_freqs(X)
