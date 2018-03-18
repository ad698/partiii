import self_consistent
import numpy as np
import sys

D1ranges = np.linspace(0.114,1.,6)

models = [self_consistent.williamsevans_selfconsistent_variableouteranisotropyhernquist(1.,i,"hernquist_we/isotropic_centre_D1_"+str(i)+"_sphr","spherical",spherical_factor=True) for i in D1ranges]

labels = [r'$D_1=$'+str(i) for i in D1ranges]

if sys.argv[1]=='build':
	for i in models:
		i.build()

if sys.argv[1]=='veldispstack':
		veldispstack(models,sys.argv[2],labels=labels)
