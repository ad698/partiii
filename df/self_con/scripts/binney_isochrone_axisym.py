import self_consistent as sc
import sys

model = sc.isochrone_selfconsistent((1.,1.),"iso_ap1_az1_axi","axisymmetric")

if sys.argv[1]=='plot':
	model.plot(True)

if sys.argv[1]=='build':
	model.build(blackhole=False)

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