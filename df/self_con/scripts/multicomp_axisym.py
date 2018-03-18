import self_consistent as sc
import sys

compts = [
			["WE",
			 #df
			 1.,5./3.,5.,1.,1.,0.378,1.,0.41,0.4,0.,0.5,
			 #me
			 1.,0.01,200.,30,4,4],
			["WE",
			 #df
			 5.,5./3.,5.,1.,1.,0.378,1.,0.41,1.,0.,0.5,
			 #me
			 1.,0.01,400.,30,4,4]
		 ]

compts = [
			["Isochrone",
			 #df
			 1.,1.,.5,1.,
			 #me
			 1.,0.01,200.,30,4,4],
			["Isochrone",
			 #df
			 1.,1.,.5,1.,
			 #me
			 1.,0.01,400.,30,4,4]
		 ]

model = sc.multicomponent_selfconsistent(compts,"multicom",ttype="axisymmetric")

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
