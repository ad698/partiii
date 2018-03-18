import self_consistent as sc
import numpy as np
import sys

model = sc.williamsevans_selfconsistent(1.,5./3.,5.,1.814,1.,0.378,1.,0.41,1.,0.,"hernquist_we_sphsph",ttype='spherical',descr=r'Williams & Evans, Hernquist, $D_0=1.814$, $q_z=1$', analytic_model=sc.Hernquist_prof)

if sys.argv[1]=='plot':
	model.plot(True)

if sys.argv[1]=='build':
	model.build()

if sys.argv[1]=='veldisp':
	model.veldisp()
	model.plot_veldisp('plots/we_disp.eps')

if sys.argv[1]=='split':
	model.split_density()

if sys.argv[1]=='proj':
	model.proj_density()

if sys.argv[1]=='genfunc':
	model.genfunc()

if sys.argv[1]=='density':
	model.density()
