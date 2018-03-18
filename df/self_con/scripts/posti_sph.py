import self_consistent as sc
import numpy as np
import sys

model = sc.posti_selfconsistent(3.,1.,4.,"May15/posti_sph_test",ttype='spherical')

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
