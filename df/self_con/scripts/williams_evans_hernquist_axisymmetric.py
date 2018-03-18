import self_consistent as sc
import numpy as np
import sys

model = sc.williamsevans_selfconsistent(1.,5./3.,5.,1.,1.,0.378*np.power((1+1.814)/(1+1.),-5./3.),1.,0.41,0.4,0.,"May15/hernquist_we_t_qz0.4_D0_1.0_axi",ttype='axisymmetric',descr=r'Williams & Evans, Hernquist, $D_0=1$, $q_z=0.4$', analytic_model=sc.Hernquist_prof)

if sys.argv[1]=='plot':
	model.plot(True)

if sys.argv[1]=='build':
	model.build()

if sys.argv[1]=='veldisp':
	model.veldisp()

if sys.argv[1]=='split':
	model.split_density()

if sys.argv[1]=='proj':
	model.proj_density()

if sys.argv[1]=='genfunc':
	model.genfunc()

if sys.argv[1]=='density':
	model.density()
