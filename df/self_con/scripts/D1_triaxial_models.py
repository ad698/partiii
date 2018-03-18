import self_consistent as sc
import sys

D1ranges=[0.05,0.114,0.2,0.3,0.4684,0.6456,0.8228,1.0]
models = [sc.williamsevans_selfconsistent_variableouteranisotropyhernquist(1.,i,"June15/hernquist_we/isotropic_centre_D1_"+str(i),"triaxial",spherical_factor=False,descr=r'Williams & Evans, Hernquist, $D_1=$'+str(i)) for i in D1ranges]

if sys.argv[1]=='plot':
	for i in models:
		i.plot()

if sys.argv[1]=='build':
	for i in models:
		i.build()

if sys.argv[1]=='split':
	for i in models:
		i.split_density()

if sys.argv[1]=='density':
	for i in models:
		i.density()
