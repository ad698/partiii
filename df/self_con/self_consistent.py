import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fnmatch
import os
import seaborn as sns
import sys
import pandas as pd
from scipy.interpolate import griddata, interp1d
from numpy.linalg import eig, inv
from subprocess import call
from StringIO import StringIO
from multiprocessing import Pool
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.patches import Ellipse

data_folder = 'data/'
plot_folder = 'plots/'

linestyles=['--',':','-.']

def Hernquist_prof(r,key='rho'):
	if(key=='name'):
		return 'Hernquist'
	elif(key=='rho'):
		return 1./r/(1+r)**3/2./np.pi
	else:
		return -1./(1+r)

def Jaffe_prof(r,key='rho'):
	if(key=='name'):
		return 'Jaffe'
	elif(key=='rho'):
		return 1./r**2/(1+r)**2
	else:
		return np.log(r/(1+r))

def Isochrone_prof(r,key='rho'):
	a = np.sqrt(1+r*r)
	if(key=='name'):
		return 'Isochrone'
	elif(key=='rho'):
		return (3*(1+a)*a*a-r*r*(1+3.*a))/4./np.pi/(1+a)**3/a/a/a
	else:
		return -1./(1+a)

def fitEllipse(x,y,length=False):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, y*y, np.ones_like(x),x*y))
    S = np.dot(D.T,D)
    C = np.zeros([4,4])
    C[0,1] = C[1,0] = 2
    C[3,3]=-1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    if(a[2]>0.):
    	a*=-1.
    l = np.sqrt(2.*(a[2]*a[3]/2./2.-a[0]*a[1]*a[2])/((0.25*a[3]*a[3]-a[0]*a[1])*(np.sqrt((a[0]-a[1])**2+a[3]*a[3])-a[0]-a[1])))
    a[3] = 0.5*np.arctan(a[3]/(a[0]-a[1]))
    if(length):
    	return a,l
    else:
    	return a

def fit_ellipses_to_contours(CS,ax=None,sb=None,length=False):
	xco = np.zeros(len(CS.collections))
	level = np.zeros(len(xco))
	paths = CS.collections
	params = np.zeros((len(paths),4+int(length)))
	for n,i in enumerate(paths):
		line = i.get_paths()[0]
		v = line.vertices
		x = v[:,0]
		y = v[:,1]
		l=0
		if(length):
			a,l = fitEllipse(x,y,length)
			a = np.append(a,l)
		else:
			a = fitEllipse(x,y)
		rr = np.linspace(0.,2.*np.pi,100)
		xx = np.sqrt(-a[2]/a[0])*np.cos(rr)
		yy = np.sqrt(-a[2]/a[1])*np.sin(rr)
		if(ax):
			ax.plot(xx,yy,color=sns.color_palette()[1],lw=0.7)
		xco[n] = np.sqrt(-a[2]/a[0])
		level[n] = np.sqrt(a[0]/a[1])
		params[n]=a
	xco=xco[xco!=0.]
	level=level[level!=0.]
	# innermost usually dodgy
	if(sb):
		sb.plot(xco,level,c='k',lw=1.)
	return params

from scipy.ndimage import gaussian_filter1d

def fit_ellipses_to_logcontours(CS,ax=None,sb=None,color='k',ls='-',label=None,ifprint=None,other_pars=None):
	xco = np.zeros(len(CS.collections))
	level = np.zeros(len(xco))
	paths = CS.collections
	for n,i in enumerate(paths):
		line = 0
		if(len(i.get_paths())>0):
			line = i.get_paths()[0]
		else:
			continue
		v = line.vertices
		x = np.exp(v[:,0])
		y = np.exp(v[:,1])
		a = fitEllipse(x,y)
		rr = np.linspace(0.,2.*np.pi,100)
		xx = np.sqrt(-a[2]/a[0])*np.cos(rr)
		yy = np.sqrt(-a[2]/a[1])*np.sin(rr)
		if(ax):
			ax.plot(xx,yy,color=sns.color_palette()[1],lw=0.7)
		xco[n] = np.sqrt(-a[2]/a[0])
		level[n] = np.sqrt(a[0]/a[1])
	xco=xco[xco!=0.]
	level=level[level!=0.]
	if(ifprint):
		print interp1d(xco, level)(ifprint*0.1)
		print interp1d(xco, level)(ifprint)
		print interp1d(xco, level)(ifprint*10.)
		print interp1d(xco, level)(ifprint*50.)
	if(other_pars):
		other_pars[1] = other_pars[1][other_pars[0]>np.nanmax(xco)]
		other_pars[0] = other_pars[0][other_pars[0]>np.nanmax(xco)]
		xco = np.concatenate((xco,other_pars[0]))
		xsort = np.argsort(xco)
		xco=xco[xsort]
		level = np.concatenate((level,other_pars[1]))
		level=level[xsort]
	if(sb):
		xnew = np.linspace(np.log(np.nanmin(xco)),np.log(np.nanmax(xco)),400)
		level=level[~np.isnan(xco)]
		xco=xco[~np.isnan(xco)]
		# x2 = np.exp(gaussian_filter1d(np.log(xco),1.))
		# y2 = gaussian_filter1d(level,0.1)
		l,=sb.plot(xco,level,c=color,ls=ls,label=label,lw=1.)
		# l,=sb.plot(x2,y2,c=color,ls=ls,label=label,lw=1.)
		# f = interp1d(np.log(xco),level,'cubic')
		# l,=sb.plot(np.exp(xnew),f(xnew),c=color,ls=ls,label=label,lw=1.)
		if(ls=='--'):
			l.set_dashes([3.,1.5])
	return xco,level

def mirror_data_spherical(data):
	data_2=data
	for p in np.arange(0.,2.*np.pi,6):
		for t in np.arange(0.,np.pi,6):
			data_c = data
			data_c['x']=data_c['r']*np.cos(p)*np.sin(t)
			data_c['y']=data_c['r']*np.sin(p)*np.sin(t)
			data_c['z']=data_c['r']*np.cos(t)
			data_c['phi']=p*np.ones(len(data_c['r']))
			data_c['theta']=p*np.ones(len(data_c['r']))
			data_2=data_2.append(data_c)
	return data_2

def spherical_interp(x,y,data):
	r = np.sqrt(x*x+y*y)
	if(r<np.min(data['r']) or r>np.max(data['r'])):
		return 0.
	f2 = interp1d(data['r'], data['rho'])
	return f2(r)[0]

class selfconsistent_model:

	def __init__(self,name,typpe='triaxial',descr=None,analytic_model=None):
		self.name = name
		self.type=typpe
		self.ln=[]
		self.descr=descr
		self.data_folder=data_folder
		self.plot_folder=plot_folder
		self.analytic_model=analytic_model

	def build(self):
		pass

	def veldisp(self):
		pass

	def plot_veldisp(self,outputfile):
		self.veldisp()
		if(self.type=='spherical'):
			vd_data = pd.read_csv(data_folder+self.name+'.xdisp',skiprows=1,sep =r'\s+',header = None,names=['r','rho','sigmarr','sigmapp'])
			plt.plot(vd_data['r'],np.sqrt(vd_data['sigmarr']/vd_data['rho']),ls='-',label=r'$\sigma_r$')
			plt.plot(vd_data['r'],np.sqrt(vd_data['sigmapp']/vd_data['rho']),ls='--',label=r'$\sigma_t$')
			plt.plot(vd_data['r'],1-vd_data['sigmapp']/vd_data['sigmarr'],ls=':',label=r'$\beta$')
			plt.legend(frameon=False)
			plt.semilogx()
			plt.xlabel(r'$r$')
			plt.savefig(outputfile,bbox_inches='tight')
		else:
			return 1

	def load_self_con_file(self,infile=None):
		if(infile==None):
			return pd.read_csv(StringIO('nan '*8),sep=' ',
		                   names = ['x','y','z','r','phi','theta','rho','pot'])
		return pd.read_csv(infile,sep=' ',
		                   names = ['x','y','z','r','phi','theta','rho','pot'])

	def bin_data_along_xy(self,xmin,xmax,nbins,with_levels=False,key='rho'):
		data = self.load_self_con_file()
		n = -1
		while(np.count_nonzero(~np.isnan(data['rho']))==0):
			data = self.load_self_con_file(self.data_folder+sorted(self.ln,key=lambda i:i[-5])[n])
			n-=1

		xi = np.logspace(xmin,xmax,nbins)
		xi=np.append(-xi[::-1],xi)
	  	yi = np.logspace(xmin,xmax,nbins)
	  	yi=np.append(-yi[::-1],yi)

	  	theta_cut = np.min(data['theta'])+1e-5

	  	x = data['x'][data['theta']<theta_cut]
	  	x=x.append(-x)
	  	x=x.append(-x)
	  	y = data['y'][data['theta']<theta_cut]
	  	y=y.append(-y)
	  	y=y.append(y)
	  	rr = data['r'][data['theta']<theta_cut]
	  	dens = data[key][data['theta']<theta_cut]
	  	if(key=='rho'):
	  		dens = np.log10(dens)
	  	f_levels = interp1d(np.log10(rr),dens-np.max(dens))
		levels = f_levels(np.log10(np.logspace(xmin,xmax,nbins+2)))[1:-1]
	  	dens=dens.append(dens)
	  	dens=dens.append(dens)

	  	zi = []
	  	if(self.type=='spherical'):
	  		zi = np.array([[spherical_interp(x,y,data) for y in yi[:,None]] for x in xi[:,None]])
	  	else:
			zi = griddata((x,y), dens-np.max(dens),
		 		(xi[None,:], yi[:,None]), method='cubic')
		if(with_levels):
			return xi,yi,zi,levels
		else:
			return xi,yi,zi

	def bin_data_along_xz(self,xmin,xmax,nbins,with_levels=False,key='rho'):
		data = self.load_self_con_file()
		n = -1
		while(np.count_nonzero(~np.isnan(data['rho']))==0):
			data = self.load_self_con_file(self.data_folder+sorted(self.ln,key=lambda i:i[-5])[n])
			n-=1
		xi = np.logspace(xmin,xmax,nbins)
		xi=np.append(-xi[::-1],xi)
	  	yi = np.logspace(xmin,xmax,nbins)
	  	yi=np.append(-yi[::-1],yi)

		phi_cut = np.min(data['phi'])+1e-4
	  	x = data['x'][data['phi']<phi_cut]
	  	x=x.append(-x)
	  	x=x.append(-x)
	  	y = np.abs(data['z'][data['phi']<phi_cut])
		y=y.append(-y)
	  	y=y.append(y)
	  	rr = data['r'][data['phi']<phi_cut]
	  	dens = data[key][data['phi']<phi_cut]
	  	if(key=='rho'):
	  		dens = np.log10(dens)
	  	f_levels = interp1d(np.log10(rr),dens-np.max(dens))
		levels = f_levels(np.log10(np.logspace(xmin,xmax,nbins+2)))[1:-1]
	  	dens=dens.append(dens)
	  	dens=dens.append(dens)
	  	if(self.type=='spherical'):
	  		zi = np.array([[spherical_interp(x,y,data) for y in yi[:,None]] for x in xi[:,None]])
	  	else:
			zi = griddata((x,y), dens-np.max(dens),
		 		(xi[None,:], yi[:,None]), method='cubic')
		if(with_levels):
			return xi,yi,zi,levels
		else:
			return xi,yi,zi

	def bin_data_along_xyaxe(self,xmin,xmax,nbins,key='rho'):
		data = self.load_self_con_file()
		n = -1
		while(np.count_nonzero(~np.isnan(data['rho']))==0):
			data = self.load_self_con_file(self.data_folder+sorted(self.ln,key=lambda i:i[-5])[n])
			n-=1
		xi = np.logspace(xmin,xmax,nbins)
	  	yi = 0.001*np.ones(len(xi))

	  	theta_cut = np.min(data['theta'])+1e-5
	  	x = data['x'][data['theta']<theta_cut]
	  	x=x.append(-x)
	  	x=x.append(-x)
	  	y = data['y'][data['theta']<theta_cut]
	  	y=y.append(-y)
	  	y=y.append(y)
	  	dens = data[key][data['theta']<theta_cut]
	  	dens=dens.append(dens)
	  	dens=dens.append(dens)
	  	zi = griddata((x,y), dens,
		 		(xi[None,:], yi[:,None]), method='cubic')
		zi2 = griddata((y,x), dens,
		 		(xi[None,:], yi[:,None]), method='cubic')
		return xi,yi,zi[0],zi2[0]

	def plot_density_and_potential_iterations(self,ax1,ax2,input_potential=True,axes_off=False):
		data = np.genfromtxt(self.data_folder+self.name+"_initpot.vis", )
		ax1.set_xlim(np.min(data.T[0])*0.9,np.max(data.T[0])*1.1)
		ax2.set_xlim(np.min(data.T[0])*0.9,np.max(data.T[0])*1.1)
		ax1.set_ylim(10e-13,10e5)

		if(input_potential):
			ax2.plot(data.T[0],data.T[1],'.',ms=3,color='k',label='Input potential')

		lnsph=[]

		for ffile in os.listdir(self.data_folder):
			if fnmatch.fnmatch(ffile,self.name+".vis"):
				data = self.load_self_con_file(self.data_folder+ffile)
				ax1.plot(data['r'],data['rho']*pow(50.,2),'.',ms=3)
				ax2.plot(data['r'],data['pot']+0.3,'.',ms=3,label='Initial potential')
		for ffile in os.listdir(self.data_folder):
			if fnmatch.fnmatch(ffile,self.name+"_it*.vis"):
				self.ln.append(ffile)
			if(fnmatch.fnmatch(ffile,self.name+"_spherical_it*.vis")
			   or fnmatch.fnmatch(ffile,self.name+"_sphr_it*.vis")):
				lnsph.append(ffile)
		kk=0
		marker=['1','+','3']
		if(self.ln==[]):
			self.ln.append(self.name+".vis")
		else:
			for j,ffile in enumerate(sorted(self.ln,key=lambda i:i[-5])):
				if(j<1 or j==len(self.ln)-1):
					data = self.load_self_con_file(self.data_folder+ffile)
					ax1.plot(data['r'],data['rho']*pow(50.,1-kk),marker[kk],ms=3,mew=0.3)
					ax2.plot(data['r'],data['pot']+0.15*(1-kk),marker[kk],ms=3,mew=0.3,label='Iteration '+str(int(ffile[-5])+1))
					kk+=1

		if(len(lnsph)>0):
			data_sph = self.load_self_con_file(self.data_folder+sorted(lnsph,key=lambda i:i[-5])[-1])
			ax1.plot(data_sph['r'],data_sph['rho'],'.',ms=3,color='k')
			ax2.plot(data_sph['r'],data_sph['pot'],'.',ms=3,color='k',label='Spherical')

		if(self.analytic_model):
			radius = np.unique(data['r'])
			andens = self.analytic_model(radius,'rho')
			anpot = self.analytic_model(radius,'pot')
			ax1.plot(radius,andens,'-',color='gray')
			ax2.plot(radius,anpot,'-',color='gray',label=self.analytic_model(0.,'name')+' profile')

		ax2.legend(frameon=False,loc='lower right',bbox_to_anchor=(1.,0.0),ncol=1)
		plt.setp(ax2.get_yticklabels()[-1],visible=False)
		ax1.semilogy()
		ax1.semilogx()
		ax2.semilogx()
		if not axes_off:
			ax1.set_xlabel(r'$r$')
		if axes_off:
			ax1.axes.get_xaxis().set_visible(False)
		ax2.set_xlabel(r'$r$')
		ax1.set_ylabel(r'$\rho$')
		ax2.set_ylabel(r'$\Phi$')

	def plot_density_split(self,filename,ax1,xmin=-9.,xmax=2.):
		data = np.genfromtxt(filename)
		data = data[data.T[0].argsort()]
		ax1.set_xlim(np.min(data.T[0])*0.9,np.max(data.T[0])*1.1)

		l, = ax1.plot(np.sqrt(3.)*data.T[0],data.T[1],'--',label='Box')
		l.set_dashes([3.25,1.7])
		l1, =ax1.plot(np.sqrt(3.)*data.T[0],data.T[2],'--',label='Short')
		l1.set_dashes([5,1.5])
		l2, = ax1.plot(np.sqrt(3.)*data.T[0],data.T[3]+data.T[4],'--',label='Long')
		l2.set_dashes([1,1])
		ax1.plot(np.sqrt(3.)*data.T[0],data.T[1]+data.T[2]+data.T[3]+data.T[4],'-',color='gray',label='Total',alpha=0.7)

		if(self.analytic_model):
			radius = np.sqrt(3.)*data.T[0]
			andens = self.analytic_model(radius,'rho')
			ax1.plot(radius,andens*3.,'-',color='k',label=self.analytic_model(0.,'name')+' profile')

		ax1.legend(frameon=False,loc='lower left',bbox_to_anchor=(0.,0.0),ncol=1)
		ax1.semilogy()
		ax1.semilogx()
		ax1.set_xlim(pow(10.,-2.),100.)
		ax1.set_ylim(pow(10.,xmin),pow(10.,xmax))
		ax1.set_xlabel(r'$r$')
		ax1.set_ylabel(r'$\rho$')

	def plot_density_potential_shapes(self,axdens_xy,axdens_xz,axpot_xy,axpot_xz,axdens_shape,axpot_shape,higherres=False,xmin=0.3,xmax=20.,text=True,labels=True,ifprint=None,other_ellipse_params=None,xmin2=0.3,xmax2=20.):
		# open last file
		nbins = 14
		xmin = np.log10(xmin)
		xmax = np.log10(xmax)
		nconts = 15

		nbins2 = 40
		xmin2 = np.log10(xmin2)
		xmax2 = np.log10(xmax2)

		## First density

		xi,yi,zi = self.bin_data_along_xy(xmin,xmax,nbins)
		xi2,yi2,zi2,levels = self.bin_data_along_xy(xmin2,xmax2,nbins2,with_levels=True)

		xtmp,ytmp,ztmp = 0,0,0
		if(higherres):
			data = np.genfromtxt(self.data_folder+self.name+'.highres')
			data = data[np.lexsort((data.T[0],data.T[1]))]
			n = int(np.sqrt(len(data.T[0])))
			x = data.T[0]
			x=np.append(x,-x)
		  	x=np.append(x,-x)
		  	print x
			xi2 = xi
			xtmp=xi
			y = data.T[1]
			y=np.append(y,-y)
		  	y=np.append(y,y)
			yi2 = yi
			ytmp=yi
			zi = np.log10(data.T[2])
			zi=np.append(zi,zi)
	  		zi=np.append(zi,zi)
			zi = griddata((x,y), zi-np.max(zi),
		 		(xi[None,:], yi[:,None]), method='cubic')
			zi2 = zi
			ztmp=np.log10(data.T[3])
			ztmp=np.append(ztmp,ztmp)
	  		ztmp=np.append(ztmp,ztmp)
			ztmp = griddata((x,y), ztmp-np.max(ztmp),
		 		(xi[None,:], yi[:,None]), method='cubic')

		CSL_1_densba = axdens_xy.contour(np.log(np.abs(xi2)),np.log(np.abs(yi2)),zi2,levels=levels,linestyle='-',linewidths=0.5,colors='k')
		CSL_2_densba = axdens_xy.contour(np.log(np.abs(xi)),np.log(np.abs(yi)),zi,nconts,linestyle='-',linewidths=0.5,colors='k')
		axdens_xy.cla()
		CS = axdens_xy.contourf(xi,yi,zi,nconts,cmap=plt.cm.Blues)
		if(labels):
			divider = make_axes_locatable(axdens_xy)
			cax = divider.append_axes("top", size="5%", pad=0.05)
			cbar = plt.colorbar(CS,cax=cax,orientation='horizontal')
			cbar.ax.tick_params(labelsize=6,pad=1.3)
		  	cbar.ax.xaxis.set_ticks_position('top')
	  	CS = axdens_xy.contour(xi,yi,-zi,nconts,linestyle='--',linewidths=0.5,colors='k')

		xi,yi,zi = self.bin_data_along_xz(xmin,xmax,nbins)
		xi2,yi2,zi2,levels = self.bin_data_along_xz(xmin2,xmax2,nbins2,with_levels=True)

	  	if(higherres):
	  		xi=xtmp
	  		xi2=xi
	  		yi=ytmp
	  		yi2=yi
	  		zi = ztmp
	  		zi2 = zi

		CSL_1_densca = axdens_xz.contour(np.log(np.abs(xi2)),np.log(np.abs(yi2)),zi2,levels=levels,linestyle='-',linewidths=0.5,colors='k')
		CSL_2_densca = axdens_xz.contour(np.log(np.abs(xi)),np.log(np.abs(yi)),zi,nconts,linestyle='-',linewidths=0.5,colors='k')
		axdens_xz.cla()
		CS = axdens_xz.contourf(xi,yi,zi,nconts,cmap=plt.cm.Blues)
		if(labels):
			divider = make_axes_locatable(axdens_xz)
			cax = divider.append_axes("top", size="5%", pad=0.05)
			cbar = plt.colorbar(CS,cax=cax,orientation='horizontal')
			cbar.ax.tick_params(labelsize=6,pad=1.3)
		  	cbar.ax.xaxis.set_ticks_position('top')
		CS = axdens_xz.contour(xi,yi,-zi,nconts,linestyle='solid',linewidths=0.5,colors='k')

		## Now potential
		xi,yi,zi = self.bin_data_along_xy(xmin,xmax,nbins,key='pot')
		xi2,yi2,zi2,levels = self.bin_data_along_xy(xmin2,xmax2,nbins2,with_levels=True,key='pot')

		CSL_1_potba = axpot_shape.contour(np.log(np.abs(xi2)),np.log(np.abs(yi2)),zi2,levels=levels,linestyle='-',linewidths=0.5,colors='k')
		CSL_2_potba = axpot_shape.contour(np.log(np.abs(xi)),np.log(np.abs(yi)),zi,nconts,linestyle='-',linewidths=0.5,colors='k')
		axpot_shape.cla()

		if(axpot_xy):
			CS = axpot_xy.contourf(xi,yi,zi,nconts,cmap=plt.cm.Blues)
		  	plt.text(0.99,0.98,r'$\Phi$',horizontalalignment='right',verticalalignment='top',transform=axpot_xy.transAxes)
		  	if(labels):
				divider = make_axes_locatable(axpot_xy)
				cax = divider.append_axes("top", size="5%", pad=0.05)
				cbar = plt.colorbar(CS,cax=cax,orientation='horizontal')
				cbar.ax.tick_params(labelsize=6,pad=1.3)
			  	cbar.ax.xaxis.set_ticks_position('top')
		  	CS = axpot_xy.contour(xi,yi,-zi,nconts,linestyle='-',linewidths=0.5,colors='k')

		xi,yi,zi = self.bin_data_along_xz(xmin,xmax,nbins,key='pot')
		xi2,yi2,zi2,levels = self.bin_data_along_xz(xmin2,xmax2,nbins2,with_levels=True,key='pot')

		CSL_1_potca = axpot_shape.contour(np.log(np.abs(xi2)),np.log(np.abs(yi2)),zi2,levels=levels,linestyle='-',linewidths=0.5,colors='k')
		CSL_2_potca = axpot_shape.contour(np.log(np.abs(xi)),np.log(np.abs(yi)),zi,nconts,linestyle='-',linewidths=0.5,colors='k')
		axpot_shape.cla()
		if(axpot_xz):
			CS = axpot_xz.contourf(xi,yi,zi,nconts,cmap=plt.cm.Blues)
			plt.text(0.99,0.98,r'$\Phi$',horizontalalignment='right',verticalalignment='top',transform=axpot_xz.transAxes)
		  	if(labels):
				divider = make_axes_locatable(axpot_xz)
				cax = divider.append_axes("top", size="5%", pad=0.05)
				cbar = plt.colorbar(CS,cax=cax,orientation='horizontal')
				cbar.ax.tick_params(labelsize=6,pad=1.3)
			  	cbar.ax.xaxis.set_ticks_position('top')
			CS = axpot_xz.contour(xi,yi,-zi,nconts,linestyle='-',linewidths=0.5,colors='k')
		ipp=None
		if(ifprint):
			ipp=ifprint

		xco1,xco2,xco3,xco4=0,0,0,0
		lev1,lev2,lev3,lev4=0,0,0,0
		op1,op2,op3,op4=None,None,None,None
		if(other_ellipse_params):
			op1=[other_ellipse_params[0],other_ellipse_params[4]]
			op2=[other_ellipse_params[1],other_ellipse_params[5]]
			op3=[other_ellipse_params[2],other_ellipse_params[6]]
			op4=[other_ellipse_params[3],other_ellipse_params[7]]

		if(self.type!='spherical'):
			xco1,lev1=fit_ellipses_to_logcontours(CSL_1_densba,sb=axdens_shape,color=sns.color_palette()[0],label=r'$(b/a)_\rho$',other_pars=op1)
			# fit_ellipses_to_logcontours(CSL_2_densba,ax=axdens_xy)

		if(self.type!='spherical'):
			xco2,lev2=fit_ellipses_to_logcontours(CSL_1_densca,sb=axdens_shape,color=sns.color_palette()[2],label=r'$(c/a)_\rho$',other_pars=op2)
			# fit_ellipses_to_logcontours(CSL_2_densca,ax=axdens_xz)

		if(self.type!='spherical'):
			xco3,lev3=fit_ellipses_to_logcontours(CSL_1_potba,sb=axpot_shape,color=sns.color_palette()[0],ls='--',label=r'$(b/a)_\Phi$',ifprint=ipp,other_pars=op3)
			# if(axpot_xy):
			# 	fit_ellipses_to_logcontours(CSL_2_potba,ax=axpot_xy)

		if(self.type!='spherical'):
			xco4,lev4=fit_ellipses_to_logcontours(CSL_1_potca,sb=axpot_shape,color=sns.color_palette()[2],ls='--',label=r'$(c/a)_\Phi$',ifprint=ipp,other_pars=op4)
			# if(axpot_xz):
				# fit_ellipses_to_logcontours(CSL_2_potca,ax=axpot_xz)

		if(text):
			plt.text(0.99,0.97,r'$\rho$',horizontalalignment='right',verticalalignment='top',transform=axdens_xy.transAxes)
			plt.text(0.99,0.98,r'$\rho$',horizontalalignment='right',verticalalignment='top',transform=axdens_xz.transAxes)

		axdens_shape.set_xlabel(r'$x$')
	  	axdens_shape.set_ylabel(r'Shape')
		axdens_shape.semilogx()
		axdens_shape.legend(frameon=False,ncol=2,fontsize=8)

	  	axpot_shape.set_xlabel(r'$x$')
	  	axpot_shape.set_ylabel(r'Shape')
		axpot_shape.semilogx()
		axpot_shape.legend(frameon=False,ncol=2,fontsize=8)

		if(labels):
			axdens_xy.set_ylabel(r'$y$')
			axdens_xy.set_xlabel(r'$x$')
		axdens_xy.set_aspect('equal')

		if(axpot_xy):
			axpot_xy.set_ylabel(r'$y$')
			axpot_xy.set_xlabel(r'$x$')
			axpot_xy.set_aspect('equal')

		if(labels):
			axdens_xz.set_ylabel(r'$z$')
			axdens_xz.set_xlabel(r'$x$')
		axdens_xz.set_aspect('equal')

		if not labels:
			plt.setp( axdens_xz.get_xticklabels(), visible=False)
			plt.setp( axdens_xy.get_xticklabels(), visible=False)
			plt.setp( axdens_xz.get_yticklabels(), visible=False)
			plt.setp( axdens_xy.get_yticklabels(), visible=False)

		if(axpot_xz):
			axpot_xz.set_ylabel(r'$z$')
			axpot_xz.set_xlabel(r'$x$')
			axpot_xz.set_aspect('equal')

		axdens_shape.set_xlim(0.05,50.)
	  	axpot_shape.set_xlim(0.05,50.)
	  	axdens_shape.set_ylim(0.,1.5)
	  	axpot_shape.set_ylim(0.,1.5)

	  	return xco1,xco2,xco3,xco4,lev1,lev2,lev3,lev4

	def plot_veldisp_paper(self,filet,axsigxx_xy,axsigyy_xy,axsigxx_xz,axsigzz_xz):
		data = np.genfromtxt(filet)
		data = data[np.lexsort((data.T[0],data.T[1]))]
		n = int(np.sqrt(len(data.T[0])))
		nconts=6
		x = np.reshape(data.T[0],(n,n))
		y = np.reshape(data.T[1],(n,n))
		z = np.reshape(data.T[3]/data.T[2],(n,n))

		CS = axsigxx_xy.contourf(x,y,z,nconts,cmap=plt.cm.Blues)
		divider = make_axes_locatable(axsigxx_xy)
		cax = divider.append_axes("right", size="6%", pad=0.05)
		cbar = plt.colorbar(CS,cax=cax)
		cbar.ax.tick_params(labelsize=6,pad=1.3)
		cbar.ax.xaxis.set_ticks_position('top')
	  	CS = axsigxx_xy.contour(x,y,z,nconts,linestyle='-',linewidths=0.5,colors='k')
		plt.text(0.96,0.98,r'$\sigma^2_{xx}$',horizontalalignment='right',verticalalignment='top',transform=axsigxx_xy.transAxes)
		axsigxx_xy.set_ylabel(r'$y$')
		axsigxx_xy.set_xlabel(r'$x$')
		axsigxx_xy.set_aspect('equal')

		x = np.reshape(data.T[0],(n,n))
		y = np.reshape(data.T[1],(n,n))
		z = np.reshape(data.T[4]/data.T[2],(n,n))

		CS = axsigyy_xy.contourf(x,y,z,nconts,cmap=plt.cm.Blues)
		divider = make_axes_locatable(axsigyy_xy)
		cax = divider.append_axes("right", size="6%", pad=0.05)
		cbar = plt.colorbar(CS,cax=cax)
		cbar.ax.tick_params(labelsize=6,pad=1.3)
		cbar.ax.xaxis.set_ticks_position('top')
		CS = axsigyy_xy.contour(x,y,z,nconts,linestyle='-',linewidths=0.5,colors='k')
		plt.text(0.96,0.98,r'$\sigma^2_{yy}$',horizontalalignment='right',verticalalignment='top',transform=axsigyy_xy.transAxes)
		axsigyy_xy.set_ylabel(r'$y$')
		axsigyy_xy.set_xlabel(r'$x$')
		axsigyy_xy.set_aspect('equal')

		x = np.reshape(data.T[0],(n,n))
		y = np.reshape(data.T[1],(n,n))
		z = np.reshape(data.T[7]/data.T[6],(n,n))

		CS = axsigxx_xz.contourf(x,y,z,nconts,cmap=plt.cm.Blues)
		divider = make_axes_locatable(axsigxx_xz)
		cax = divider.append_axes("right", size="6%", pad=0.05)
		cbar = plt.colorbar(CS,cax=cax)
		cbar.ax.tick_params(labelsize=6,pad=1.3)
	  	CS = axsigxx_xz.contour(x,y,z,nconts,linestyle='-',linewidths=0.5,colors='k')
		plt.text(0.96,0.98,r'$\sigma^2_{xx}$',horizontalalignment='right',verticalalignment='top',transform=axsigxx_xz.transAxes)
		axsigxx_xz.set_ylabel(r'$z$')
		axsigxx_xz.set_xlabel(r'$x$')
		axsigxx_xz.set_aspect('equal')

		x = np.reshape(data.T[0],(n,n))
		y = np.reshape(data.T[1],(n,n))
		z = np.reshape(data.T[8]/data.T[6],(n,n))

		CS = axsigzz_xz.contourf(x,y,z,nconts,cmap=plt.cm.Blues)
		divider = make_axes_locatable(axsigzz_xz)
		cax = divider.append_axes("right", size="6%", pad=0.05)
		cbar = plt.colorbar(CS,cax=cax)
		cbar.ax.tick_params(labelsize=6,pad=1.3)
	  	CS = axsigzz_xz.contour(x,y,z,nconts,linestyle='-',linewidths=0.5,colors='k')
		plt.text(0.96,0.98,r'$\sigma^2_{zz}$',horizontalalignment='right',verticalalignment='top',transform=axsigzz_xz.transAxes)
		axsigzz_xz.set_ylabel(r'$z$')
		axsigzz_xz.set_xlabel(r'$x$')
		axsigzz_xz.set_aspect('equal')

	def plot_proj_paper(self,filet,a):

		data = np.genfromtxt(filet)
		nconts = 6
		x = np.linspace(-np.sqrt(2.),np.sqrt(2.),100)
		y = x
		z = griddata((data.T[0],data.T[1]),np.log10(data.T[2])-np.log10(np.max(data.T[2]))+0.01,(x[None,:], y[:,None]), method='cubic')

		CS = a.contourf(x,y,z,nconts,cmap=plt.cm.Blues)
		CSa = a.contour(x,y,-z,nconts,linestyle='-',linewidths=0.5,colors='dimgrey')
		f = fit_ellipses_to_contours(CSa,None,None,True)
		for i in f.T[3][:-1]:
			plt.plot(x,np.tan(i)*x,color='k',lw=0.6)
		print (f.T[3][4]-f.T[3][0])/(f.T[4][4]-f.T[4][0])*180./np.pi
		divider = make_axes_locatable(a)
		cax = divider.append_axes("right", size="6%", pad=0.05)
		cbar = plt.colorbar(CS,cax=cax)
		cbar.ax.tick_params(labelsize=6,pad=1.3)
		cbar.ax.xaxis.set_ticks_position('top')
		plt.text(0.96,0.95,r'$\rho_{\rm proj}$',horizontalalignment='right',verticalalignment='top',transform=a.transAxes)
		a.set_ylabel(r'$y$')
		a.set_xlabel(r'$x$')
		a.set_aspect('equal')

	def velocity_ellipses(self,x,y,dens,sigxx,sigyy,sigxy,width,ax1,ax2,size=.5,labely=r'$y$',xcut=[0.3,1.],xnum=0.555767):

	    # Draw faint polar lines
	    xi = np.linspace(0.,width)
	    for i in np.linspace(0.,np.pi/2.,15):
	        ax1.plot(xi,np.tan(i)*xi,color="lightgray",zorder=0)

	    # x-y
	    mask = (x<xcut[1])*(y<xcut[1])*(x>xcut[0])*(y>xcut[0])*(x!=xnum)*(y!=xnum)
	    x2 = x[mask]
	    y2 = y[mask]
	    dens2 = dens[mask]
	    sigxx2 = sigxx[mask]
	    sigyy2 = sigyy[mask]
	    sigxy2 = sigxy[mask]

	    def eigsorted(cov):
	        vals, vecs = np.linalg.eigh(cov)
	        order = vals.argsort()[::-1]
	        return vals[order], vecs[:,order]

	    diagonals_xy = [eigsorted(np.array([[i,j],[j,k]])) for i,j,k in zip(sigxx2,sigxy2,sigyy2)]

	    def sizex(xx,yy):
	    	return size

	    ells_xy = [Ellipse(xy=(xx,yy),
	                    width= sizex(xx,yy),
	                    height=sizex(xx,yy)*np.sqrt(j[0][1]/j[0][0]),
	                    angle=np.degrees(np.arctan2(*j[1][:,0][::-1])),zorder=2)
	                    for xx,yy,j in zip(x2,y2,diagonals_xy)]

	    ax1.set_aspect('equal')
	    ax1.set_xlabel(r'$x$')
	    ax1.set_ylabel(labely)
	    ax1.set_xlim(0,width)
	    ax1.set_ylim(0,width)

	    # Find max and min velocity dispersions
	    maxi,mini = 0.,1e5
	    for dd,d1 in zip(dens2,diagonals_xy):
	        maxic1 = np.sqrt(np.max(d1[0])/dd)
	        if(maxic1>maxi):
	            maxi = maxic1
	        elif(maxic1<mini):
	            mini = maxic1

	    print mini,maxi
	    # Colorbar
	    CMAP = sns.blend_palette(sns.hls_palette(7),as_cmap=True)
	    cNorm = colors.Normalize(vmin=mini,vmax=maxi)
	    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=CMAP)
	    scalarMap._A=[]
	    divider = make_axes_locatable(ax1)
	    cax = divider.append_axes("right", size="6%", pad=0.05)
	    cbar = plt.colorbar(scalarMap,cax=cax)
	    cbar.ax.tick_params(labelsize=6,pad=1.3)

	    plt.text(0.98,0.95,r'$\sigma_{\rm maj}$',horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes,fontsize=8)

	    for dd,e,d in zip(dens2,ells_xy,diagonals_xy):
	        ax1.add_artist(e)
	        e.set_clip_box(ax1.bbox)
	        e.set_facecolor(scalarMap.to_rgba(np.sqrt(np.max(d[0])/dd)))

		diagonals_xy = [eigsorted(np.array([[i,j],[j,k]])) for i,j,k in zip(sigxx,sigxy,sigyy)]
	    offset_xy = [ 180./np.pi*np.arccos(
	                np.abs(np.dot(d[1][:,0],np.array([xx,yy])))
	                /np.sqrt(xx**2+yy**2))
	                for xx,yy,d in zip(x,y,diagonals_xy)]
	    sh = int(np.sqrt(len(x)))
	    X = np.reshape(x,(sh,sh))
	    Y = np.reshape(y,(sh,sh))
	    offset_xy = np.reshape(offset_xy,(sh,sh))
	    CC = ax2.contourf(X,Y,offset_xy,cmap=plt.cm.Blues,norm=matplotlib.colors.LogNorm(),levels=[0.1,pow(10.,-0.5),1.,pow(10.,0.5),10.,pow(10.,1.5),pow(10.,2.)])
	    CC_l = ax2.contour(X,Y,offset_xy,norm=matplotlib.colors.LogNorm(),levels=[0.1,pow(10.,-0.5),1.,pow(10.,0.5),10.,pow(10.,1.5),pow(10.,2.)],colors='k',linestyle='-',linewidths=0.5,)
	    ax2.set_aspect('equal')
	    ax2.set_xlabel(r'$x$')
	    ax2.set_ylabel(labely)
	    ax2.set_xlim(0.,width)
	    ax2.set_ylim(0.,width)

	    divider = make_axes_locatable(ax2)
	    cax = divider.append_axes("right", size="6%", pad=0.05)
	    cbar = plt.colorbar(CC,cax=cax)
	    cbar.ax.tick_params(labelsize=6,pad=1.3)
	    plt.text(0.98,0.95,r'$\alpha/{\rm deg}$',horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes,fontsize=8)

	def add_subplot_axes(self,ax,rect,axisbg='w'):
	    fig = plt.gcf()
	    box = ax.get_position()
	    width = box.width
	    height = box.height
	    inax_position  = ax.transAxes.transform(rect[0:2])
	    transFigure = fig.transFigure.inverted()
	    infig_position = transFigure.transform(inax_position)
	    x = infig_position[0]
	    y = infig_position[1]
	    width *= rect[2]
	    height *= rect[3]  # <= Typo was here
	    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
	    x_labelsize = subax.get_xticklabels()[0].get_size()
	    y_labelsize = subax.get_yticklabels()[0].get_size()
	    x_labelsize *= rect[2]**0.5
	    y_labelsize *= rect[3]**0.5
	    subax.xaxis.set_tick_params(labelsize=x_labelsize)
	    subax.yaxis.set_tick_params(labelsize=y_labelsize)
	    return subax

	def plot(self,for_paper=True):

		self.data_folder = data_folder+'/'.join(self.name.split('/')[:-1])+'/'
		self.plot_folder = plot_folder+'/'.join(self.name.split('/')[:-1])+'/'
		self.name = self.name.split('/')[-1]

		if not for_paper:
			f,a = plt.subplots(4,2,figsize=[8.,12.])
			plt.subplots_adjust(wspace=0.3,hspace=0.3)
			self.plot_density_and_potential_iterations(a[0][0],a[0][1])

			self.plot_density_potential_shapes(a[1][0],a[2][0],a[1][1],a[2][1],a[3][0],a[3][1],False,xmin=0.01,xmax=5.)
		  	if(self.descr):
		  		plt.text(0.,1.02,self.descr,horizontalalignment='left',verticalalignment='bottom',transform=a[0][0].transAxes)

  			plt.savefig(self.plot_folder+self.name+'_plot.pdf',bbox_inches='tight')
			plt.clf()
		else:

			f,a = plt.subplots(2,1,figsize=[3.32,4.])
			plt.subplots_adjust(wspace=0.3,hspace=0.)
			self.plot_density_and_potential_iterations(a[0],a[1],input_potential=False,axes_off=True)
		  	if(self.descr):
		  		plt.text(0.,1.02,self.descr,horizontalalignment='left',verticalalignment='bottom',transform=a[0].transAxes,fontsize=9)
			plt.savefig(self.plot_folder+self.name+'_plot_converg.pdf',bbox_inches='tight')
			plt.clf()

			f,a = plt.subplots(1,3,figsize=[8.,2.1])
			plt.subplots_adjust(wspace=0.3)
			x1,x2,x3,x4,l1,l2,l3,l4=self.plot_density_potential_shapes(a[0],a[1],None,None,a[2],a[2],False,ifprint=None,xmin2=.05,xmax2=100.)
			ax_in1 = self.add_subplot_axes(a[0],[0.044,0.,0.4,0.4])
			ax_in2 = self.add_subplot_axes(a[1],[0.044,0.,0.4,0.4])
			self.plot_density_potential_shapes(ax_in1,ax_in2,None,None,a[2],a[2],False,0.02,1.,False,False,ifprint=None,other_ellipse_params=[x1,x2,x3,x4,l1,l2,l3,l4],xmin2=0.01,xmax2=0.05)
		  	if(self.descr):
		  		plt.text(0.,1.15,self.descr,horizontalalignment='left',verticalalignment='bottom',transform=a[0].transAxes,fontsize=9)
			plt.savefig(self.plot_folder+self.name+'_plot_dens.pdf',bbox_inches='tight')
			plt.clf()
			# return
			# f,a = plt.subplots(1,4,figsize=[8.,2.1])
			# plt.subplots_adjust(wspace=0.45)
			# self.plot_veldisp_paper(self.data_folder+self.name+'.xdisp',a[0],a[1],a[2],a[3])
			# for i in a:
			# 	i.set_ylim(0.,5.)
			# 	i.set_xlim(0.,5.)
		 #  	if(self.descr):
		 #  		plt.text(0.,1.05,self.descr,horizontalalignment='left',verticalalignment='bottom',transform=a[0].transAxes,fontsize=9)
			# plt.savefig(self.plot_folder+self.name+'_plot_veldisp.pdf',bbox_inches='tight')
			# plt.clf()

			f,a = plt.subplots(1,1,figsize=[3.32,2.])
			plt.subplots_adjust(wspace=0.3,hspace=0.)
			xmin,xmax=-11.,2.
			if(self.descr[0]=='B'):
				xmax=0.
			self.plot_density_split(self.data_folder+self.name+".split",a,xmin,xmax)
		  	if(self.descr):
		  		plt.text(0.,1.02,self.descr,horizontalalignment='left',verticalalignment='bottom',transform=a.transAxes,fontsize=9)
			plt.savefig(self.plot_folder+self.name+'_plot_split.pdf',bbox_inches='tight')
			plt.clf()

			f,a = plt.subplots(3,4,figsize=[8.,5.])
			plt.subplots_adjust(wspace=0.55)

			self.plot_veldisp_paper(self.data_folder+self.name+'.xdisp',a[0][0],a[0][1],a[0][2],a[0][3])
			for i in a[0]:
				i.set_ylim(0.,5.)
				i.set_xlim(0.,5.)
		  	if(self.descr):
		  		plt.text(0.,1.05,self.descr,horizontalalignment='left',verticalalignment='bottom',transform=a[0][0].transAxes,fontsize=9)
			data = np.genfromtxt(self.data_folder+self.name+'.xdisp')
			data = data[np.lexsort((data.T[0],data.T[1]))]
			self.velocity_ellipses(data.T[0],data.T[1],data.T[2],data.T[3],data.T[4],data.T[5],5.,a[1][0],a[1][1],xcut=[0.3,5.])
			self.velocity_ellipses(data.T[0],data.T[1],data.T[6],data.T[7],data.T[8],data.T[9],5.,a[1][2],a[1][3],labely=r'$z$',xcut=[0.3,5.])
			self.velocity_ellipses(data.T[0],data.T[1],data.T[2],data.T[3],data.T[4],data.T[5],1.,a[2][0],a[2][1],xcut=[0.06,5.],xnum=0.13102,size=.1)
			self.velocity_ellipses(data.T[0],data.T[1],data.T[6],data.T[7],data.T[8],data.T[9],1.,a[2][2],a[2][3],labely=r'$z$',xcut=[0.06,5.],xnum=0.13102,size=.1)
		  	# if(self.descr):
		  	# 	plt.text(0.,1.05,self.descr,horizontalalignment='left',verticalalignment='bottom',transform=a[0].transAxes,fontsize=9)
			plt.savefig(self.plot_folder+self.name+'_plot_velellips.pdf',bbox_inches='tight')
			plt.clf()


			f,a = plt.subplots(1,1,figsize=[3.32,2.])
			plt.subplots_adjust(wspace=0.3,hspace=0.)
			self.plot_proj_paper(self.data_folder+self.name+".proj",a)
		  	if(self.descr):
		  		plt.text(0.,1.02,self.descr[:-20],horizontalalignment='left',verticalalignment='bottom',transform=a.transAxes,fontsize=9)
			plt.savefig(self.plot_folder+self.name+'_plot_proj.pdf',bbox_inches='tight')
			plt.clf()


class a1a2a3_selfconsistent(selfconsistent_model):
	def __init__(self,a,J0,p,q,s,name,typpe='triaxial',profile='NFW',descr=None, analytic_model=None):
		self.a = a
		self.J0=J0
		self.p=p
		self.q=q
		self.s=s
		self.name = name
		self.type=typpe
		self.profile=profile
		self.ln=[]
		self.descr=descr
		self.analytic_model=analytic_model

	def build(self):
		proc = call(["nice -n 15 .././build_self_consistent.exe Double "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" "+str(self.a[2])+" "+str(self.J0)+" "+str(self.p)+" "+str(self.q)+" "+str(self.s)+" "+self.profile],shell=True)

	def veldisp(self):
		proc = call(["VELDISP=1 .././build_self_consistent.exe Double "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" "+str(self.a[2])+" "+str(self.J0)+" "+str(self.p)+" "+str(self.q)+" "+str(self.s)+" "+self.profile],shell=True)


class williamsevans_selfconsistent(selfconsistent_model):
	def __init__(self,J0,lmbda,mu,D0,D1,Sa,Sg,Jb,qz,b0,name,ttype='spherical',descr='Williams & Evans',analytic_model=None):
		self.J0=J0
		self.lmbda=lmbda
		self.mu=mu
		self.D0=D0
		self.D1=D1
		# if ttype!='spherical':
		# 	self.D0/=2.
		# 	self.D1/=2.
		self.Sa=Sa
		self.Sg=Sg
		self.Jb=Jb
		self.b0=b0
		self.qz=qz
		self.name = name
		self.type=ttype
		self.ln=[]
		self.descr=descr
		self.analytic_model=analytic_model

	def build(self,blackhole=False):
		envvar = ""
		if(blackhole):
			envvar="BLACKHOLE=1 "
		proc = call([envvar+"nice -n 15 .././build_self_consistent.exe WE "+self.type+" "+self.name+" "+str(self.J0)+" "+str(self.lmbda)+" "+str(self.mu)+" "+str(self.D0)+" "+str(self.D1)+" "+str(self.Sa)+" "+str(self.Sg)+" "+str(self.Jb)+" "+str(self.qz)+" "+str(self.b0)],shell=True)

	def veldisp(self):
		proc = call(["VELDISP=1 .././build_self_consistent.exe WE "+self.type+" "+self.name+" "+str(self.J0)+" "+str(self.lmbda)+" "+str(self.mu)+" "+str(self.D0)+" "+str(self.D1)+" "+str(self.Sa)+" "+str(self.Sg)+" "+str(self.Jb)+" "+str(self.qz)+" "+str(self.b0)],shell=True)

	def split_density(self):
		proc = call(["SPLIT=1 .././build_self_consistent.exe WE "+self.type+" "+self.name+" "+str(self.J0)+" "+str(self.lmbda)+" "+str(self.mu)+" "+str(self.D0)+" "+str(self.D1)+" "+str(self.Sa)+" "+str(self.Sg)+" "+str(self.Jb)+" "+str(self.qz)+" "+str(self.b0)],shell=True)

	def proj_density(self):
		proc = call(["PROJ=1 .././build_self_consistent.exe WE "+self.type+" "+self.name+" "+str(self.J0)+" "+str(self.lmbda)+" "+str(self.mu)+" "+str(self.D0)+" "+str(self.D1)+" "+str(self.Sa)+" "+str(self.Sg)+" "+str(self.Jb)+" "+str(self.qz)+" "+str(self.b0)],shell=True)

	def highres_density(self):
		proc = call(["HIGHRES=1 .././build_self_consistent.exe WE "+self.type+" "+self.name+" "+str(self.J0)+" "+str(self.lmbda)+" "+str(self.mu)+" "+str(self.D0)+" "+str(self.D1)+" "+str(self.Sa)+" "+str(self.Sg)+" "+str(self.Jb)+" "+str(self.qz)+" "+str(self.b0)],shell=True)

	def genfunc(self):
		proc = call(["GENFUNC=1 .././build_self_consistent.exe WE "+self.type+" "+self.name+" "+str(self.J0)+" "+str(self.lmbda)+" "+str(self.mu)+" "+str(self.D0)+" "+str(self.D1)+" "+str(self.Sa)+" "+str(self.Sg)+" "+str(self.Jb)+" "+str(self.qz)+" "+str(self.b0)],shell=True)

	def density(self):
		proc = call(["DENSITY=1 .././build_self_consistent.exe WE "+self.type+" "+self.name+" "+str(self.J0)+" "+str(self.lmbda)+" "+str(self.mu)+" "+str(self.D0)+" "+str(self.D1)+" "+str(self.Sa)+" "+str(self.Sg)+" "+str(self.Jb)+" "+str(self.qz)+" "+str(self.b0)],shell=True)

class posti_selfconsistent(selfconsistent_model):
	def __init__(self,J0,alpha,beta,name,ttype='spherical'):
		self.J0=J0
		self.alpha=alpha
		self.beta=beta
		self.name = name
		self.type=ttype
		self.ln=[]

	def build(self):
		proc = call([".././build_self_consistent.exe PB "+self.type+" "+self.name+" "+str(self.J0)+" "+str(self.alpha)+" "+str(self.beta)],shell=True)


class isochrone_selfconsistent(selfconsistent_model):
	def __init__(self,a,name,typpe='triaxial'):
		self.a = a
		self.name = name
		self.type=typpe
		self.ln=[]
		self.analytic_model=Isochrone_prof
		self.descr = r'Binney flattened isochrone, $\alpha_{\phi 0}='+str(self.a[0])+r',\,\alpha_{z0}='+str(self.a[1])+r'$'

	def build(self,blackhole=False):
		envvar = ""
		if(blackhole):
			envvar="BLACKHOLE=1 "
		proc = call([envvar+"nice -n 15 .././build_self_consistent.exe Isochrone "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" 1. 1. "],shell=True)

	def veldisp(self):
		proc = call(["VELDISP=1 .././build_self_consistent.exe Isochrone "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" 1. 1. "],shell=True)

	def split_density(self):
		proc = call(["SPLIT=1 .././build_self_consistent.exe Isochrone "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" 1. 1. "],shell=True)

	def proj_density(self):
		proc = call(["PROJ=1 .././build_self_consistent.exe Isochrone "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" 1. 1. "],shell=True)

	def highres_density(self):
		proc = call(["HIGHRES=1 .././build_self_consistent.exe Isochrone "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" 1. 1. "],shell=True)

	def genfunc(self):
		proc = call(["GENFUNC=1 .././build_self_consistent.exe Isochrone "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" 1. 1. "],shell=True)

	def density(self):
		proc = call(["DENSITY=1 .././build_self_consistent.exe Isochrone "+self.type+" "+self.name+" "+str(self.a[0])+" "+str(self.a[1])+" 1. 1. "],shell=True)

def a1a2a3_equal_one_models(ba=1.,ca=1.,J0=1.,p=3.,q=0.5,name="tmp",ttype="triaxial"):
	return  a1a2a3_selfconsistent([np.power(ca*ba,-1./3.),np.power(ba*ba/ca,1./3.),np.power(ca*ca/ba,1./3.)],J0,p,q,name,ttype)

def AandC_models(A=0.55,C=1.,J0=1.,p=10./3.,q=5./3.,s=2.,name="tmp",ttype="triaxial",profile="Hernquist",spherical_factor=1.,model_type="Hernquist",analytic_model=None):
	return a1a2a3_selfconsistent([A*spherical_factor*np.power(C,2./3.),A*np.power(C,-1./3.),A*np.power(C,-1./3.)],J0,p,q,s,name,ttype,profile,r'$A='+str(A)+r', C='+str(C)+r'$, '+model_type,analytic_model)

def AandCandD_models(A=0.55,C=1.,D=1.,J0=1.,p=10./3.,q=5./3.,s=2.,name="tmp",ttype="triaxial",profile="Hernquist",spherical_factor=1.,model_type="Hernquist",analytic_model=None):
	return a1a2a3_selfconsistent([A*spherical_factor*np.power(C,2./3.)*np.power(D,-1./3.),A*np.power(C*D,-1./3.),A*np.power(D,2./3.)*np.power(C,-1./3.)],J0,p,q,s,name,ttype,profile,r'$A='+str(A)+r', C='+str(C)+r', D='+str(D)+r'$, '+model_type,analytic_model)


class williamsevans_selfconsistent_variableouteranisotropyhernquist(williamsevans_selfconsistent):
	def __init__(self,J0,D1,name,ttype='spherical',spherical_factor=False,descr='Williams & Evans, Hernquist'):
		self.J0=J0
		self.lmbda=5./3.
		self.mu=5.
		self.D0=1.814
		self.D1=D1
		# if not spherical_factor:
		# 	self.D0/=2.
		# 	self.D1/=2.
		self.Sa=0.378
		self.Sg=np.power(2./(1+self.D1),-self.mu)
		self.Jb=0.41
		self.b0=0.
		self.qz=1.
		self.name = name
		self.type=ttype
		self.analytic_model=Hernquist_prof
		self.ln=[]
		self.descr=descr

class williamsevans_selfconsistent_variableinneranisotropyhernquist(williamsevans_selfconsistent):
	def __init__(self,J0,D0,name,ttype='spherical',spherical_factor=False,descr='Williams & Evans, Hernquist'):
		self.J0=J0
		self.lmbda=5./3.
		self.mu=5.
		self.D0=D0
		self.D1=1.
		if not spherical_factor:
			self.D0/=2.
			self.D1/=2.
		self.Sa=0.378*np.power((1+1.814)/(1+self.D0),-self.lmbda)
		self.Sg=1.
		self.Jb=0.41
		self.b0=0.
		self.qz=1.
		self.name = name
		self.type=ttype
		self.analytic_model=Hernquist_prof
		self.ln=[]
		self.descr=descr

class williamsevans_selfconsistent_variableinneranisotropy_method2_hernquist(williamsevans_selfconsistent):
	def __init__(self,J0,b0,name,ttype='spherical',spherical_factor=False,descr='Williams & Evans, Hernquist'):
		self.J0=J0
		self.lmbda=5./3.
		self.mu=5.
		self.D0=1.814
		self.D1=.7
		if not spherical_factor:
			self.D0/=2.
			self.D1/=2.
		self.Sa=0.378
		self.Sg=1.
		self.Jb=0.41
		self.b0=b0
		self.qz=1.
		self.name = name
		self.type=ttype
		self.analytic_model=Hernquist_prof
		self.ln=[]
		self.descr=descr

class multicomponent_selfconsistent(selfconsistent_model):
	def __init__(self,compts,name,ttype='spherical',descr='MultiComponent'):
		self.name = name
		self.type=ttype
		self.descr=descr
		self.configfile = data_folder+name+".config"
		with open (self.configfile,'wt') as f:
			for i in compts:
				for ee in i:
					f.write("%s " % ee)
				f.write("\n")

	def build(self,blackhole=False):
		envvar = ""
		if(blackhole):
			envvar="BLACKHOLE=1 "
		proc = call([envvar+"nice -n 15 .././build_multi_self_consistent.exe "+self.type+" "+self.name+" "+self.configfile],shell=True)

	def veldisp(self):
		proc = call(["VELDISP=1 .././build_multi_self_consistent.exe "+self.type+" "+self.name+" "+self.configfile],shell=True)

	def split_density(self):
		proc = call(["SPLIT=1 .././build_multi_self_consistent.exe "+self.type+" "+self.name+" "+self.configfile],shell=True)

	def proj_density(self):
		proc = call(["PROJ=1 .././build_multi_self_consistent.exe "+self.type+" "+self.name+" "+self.configfile],shell=True)

	def highres_density(self):
		proc = call(["HIGHRES=1 .././build_multi_self_consistent.exe "+self.type+" "+self.name+" "+self.configfile],shell=True)

	def genfunc(self):
		proc = call(["GENFUNC=1 .././build_multi_self_consistent.exe "+self.type+" "+self.name+" "+self.configfile],shell=True)

	def density(self):
		proc = call(["DENSITY=1 .././build_multi_self_consistent.exe "+self.type+" "+self.name+" "+self.configfile],shell=True)

import os.path
from matplotlib.ticker import MaxNLocator

def density_comp(file1,file2,plotname,text):
	f,a = plt.subplots(2,1,figsize=[3.32,2.6])
	plt.subplots_adjust(hspace=0)
	data1 = np.genfromtxt(data_folder+file1)
	data1.T[1]=data1.T[1][np.argsort(data1.T[0])]
	data1.T[0]=np.sort(data1.T[0])
	a[0].plot(data1.T[0],data1.T[1],'k-',lw=0.5,label='Fudge')
	data2 = np.genfromtxt(data_folder+file2)
	data2.T[1]=data2.T[1][np.argsort(data2.T[0])]
	data2.T[0]=np.sort(data2.T[0])
	l, = a[0].plot(data2.T[0],data2.T[1],'r--',lw=0.5,label='Generating function')
	a[0].legend(frameon=False,loc='lower left')
	l.set_dashes([3.,1.5])
	a[1].yaxis.set_major_locator(MaxNLocator(prune='upper'))
	a[1].plot(data1.T[0],(data1.T[1]-data2.T[1])/data2.T[1],'k-',lw=0.5)
	a[1].set_xlabel(r'$r$')
	a[1].set_ylabel(r'$\Delta\rho/\rho$')
	a[0].set_ylabel(r'$\rho$')
	a[0].set_xlim(0.01,30.)
	a[1].set_xlim(0.01,30.)
	a[0].semilogx()
	a[1].semilogx()
	a[0].semilogy()

	# plt.setp(a[1].get_yticklabels()[-2],visible=False)
	a[0].axes.get_xaxis().set_visible(False)
	plt.text(0.,1.02,text,horizontalalignment='left',verticalalignment='bottom',transform=a[0].transAxes,fontsize=8)
	plt.savefig(plot_folder+plotname+'density_comp.pdf',bbox_inches='tight')

def veldispstack(models,plot_name,labels):
	f,a = plt.subplots(2,1,figsize=[3.32,4.])
	plt.subplots_adjust(hspace=0.3)
	for i,mod in enumerate(models):
		if(mod.type=='spherical'):
			fname = data_folder+mod.name+'.xdisp'
			if not os.path.isfile(fname):
				mod.veldisp()
			vd_data = pd.read_csv(fname,skiprows=1,sep =r'\s+',header = None,names=['r','rho','sigmarr','sigmapp'])
			if(i==len(models)-1):
				color='r'
			else:
				color='k'
			a[0].plot(vd_data['r'],1-vd_data['sigmapp']/vd_data['sigmarr'],color=color,lw=0.5)
			n=len(vd_data['r'])
			if(i<len(models)-1):
				a[0].text(vd_data['r'][n-2],(1-vd_data['sigmapp']/vd_data['sigmarr'])[n-2],"%.2f" % float(labels[i][6:]),horizontalalignment='right',verticalalignment='bottom',fontsize=6,color=color)
	# plt.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.05),ncol=3)
	dd = np.genfromtxt("ba_with_D1.txt")
	xnew = np.linspace(dd.T[0].min(),dd.T[0].max(),300)
	f = interp1d(dd.T[0],dd.T[2],'cubic')
	a[1].plot(xnew,f(xnew),'k-',label='$r=1$')
	xnew = np.linspace(dd.T[0].min(),dd.T[0].max(),300)
	f = interp1d(dd.T[0],dd.T[3],'cubic')
	l,=a[1].plot(xnew,f(xnew),ls='--',color=sns.color_palette()[2],label='$r=10$')
	l.set_dashes([3.,1.5])
	f = interp1d(dd.T[0],dd.T[4],'cubic')
	l,=a[1].plot(xnew,f(xnew),ls='--',color=sns.color_palette()[0],label='$r=50$')
	l.set_dashes([5,1.5])
	a[1].legend(frameon=False,loc='lower right')
	xnew = np.linspace(0.,1.,300)
	l,=a[1].plot(xnew,np.ones(len(xnew)),color='gray',ls='--',alpha=0.5)
	l.set_dashes([1,1])
	a[0].semilogx()
	a[0].set_xlabel(r'$r$')
	a[0].set_ylabel(r'$\beta$')
	a[1].set_xlabel(r'$D_1$')
	a[1].set_ylabel(r'$(b/a)_\Phi$')
	plt.savefig(plot_name,bbox_inches='tight')

modelsiso = [isochrone_selfconsistent((0.7,1.4),"iso_ap0.7_az1.4","triaxial"),isochrone_selfconsistent((1.0,1.4),"iso_ap1.0_az1.4","triaxial"),isochrone_selfconsistent((1.2,1.4),"iso_ap1.2_az1.4","triaxial"),isochrone_selfconsistent((1.3,1.6),"iso_ap1.3_az1.6","triaxial")]

if __name__=='__main__':

	# Grid
	ranges = np.arange(1.,1.05,0.1)
	# ranges = np.arange(0.7,1.2,0.1)
	# ranges = np.arange(1.2,1.7,0.1)
	# ranges = np.arange(1.7,2.2,0.1)

	# models=[selfconsistent_model([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,10./3.,5./3.,"p3.33q1.66J01_"+str(i)+"_1_1_sphr","spherical","Hernquist") for i in ranges]

	models=[a1a2a3_selfconsistent([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,3.,.5,1.,"p3q05J01_"+str(i)+"_1_1","triaxial") for i in ranges]

	# models=[a1a2a3_selfconsistent([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,10./3.,5./3.,"p3.33q1.66J01_"+str(i)+"_1_1_sphr","spherical","Hernquist") for i in ranges]

	# models = [isochrone_selfconsistent((i,i),"iso_ap"+str(i)+"_az"+str(i),"triaxial") for i in ranges]

	jaffe_we_models = [williamsevans_selfconsistent(1.,2.,5.,1.8,1.,1.,1.,0.69,0.,1.,"jaffe_we")]

	hernquist_we_models = [williamsevans_selfconsistent(1.,5./3.,5.,1.814,1.,0.378,1.,0.41,0.,1.,"hernquist_we_t",ttype='triaxial')]

	# hernquist_we_triaxial_models = [williamsevans_selfconsistent(1.,5./3.,5.,1.814,1.,0.378,1.,0.41,"hernquist_we_t","triaxial")]

	# models = [posti_selfconsistent(1.,1.,4.,"hernquist_pb")]

	# models=[a1a2a3_selfconsistent([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,3.,2.,2.,"p3q2s2J01_"+str(i)+"_1_1","triaxial") for i in ranges]

	# models=[a1a2a3_selfconsistent([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,3.,2.,2.,"p3q2s2J01_"+str(i)+"_1_1_sphr","spherical") for i in ranges]

	# models=[a1a2a3_selfconsistent([.57*2.*np.power(i,2./3.),.57*np.power(i,-1./3.),.57*np.power(i,-1./3.)],1.,10./3.,5./3.,2.,"p3.33q1.66s2J01_"+str(i)+"_.57_.57_sphr_2","spherical","Hernquist") for i in ranges]

	ranges = [0.86,] # 0.2,0.3,0.5,0.8,1.0,1.2,1.5,2.]

	# models = [AandC_models(A=0.57,C=i,J0=1.,p=10./3.,q=5./3.,s=2.,name="hernquist_AC_models/A0.57C"+str(i)+"_hernquist",ttype="triaxial",profile="Hernquist",spherical_factor=1.,analytic_model=Hernquist_prof) for i in ranges]

	# models = [AandC_models(A=0.57,C=i,J0=1.,p=10./3.,q=5./3.,s=2.,name="hernquist_AC_models/A0.57C"+str(i)+"_hernquist_spherical",ttype="spherical",profile="Hernquist",spherical_factor=2.) for i in ranges]

	# models = [AandC_models(A=0.73,C=i,J0=1.,p=3.,q=2.,s=2.,name="jaffe_AC_models/A0.73C"+str(i)+"_jaffe",ttype="triaxial",profile="Hernquist",spherical_factor=1.,model_type="Jaffe",analytic_model=Jaffe_prof) for i in ranges]

	# models = [AandC_models(A=0.73,C=i,J0=1.,p=3.,q=2.,s=2.,name="jaffe_AC_models/A0.73C"+str(i)+"_jaffe_spherical",ttype="spherical",profile="Hernquist",spherical_factor=2.,model_type="Jaffe") for i in ranges]

	D1ranges = np.linspace(0.114,1.,6)
	D1ranges=np.append(D1ranges,0.2)

	hernquist_variable_outer_anis = [williamsevans_selfconsistent_variableouteranisotropyhernquist(1.,i,"hernquist_we/isotropic_centre_D1_"+str(i)+"_sphr","spherical",spherical_factor=True) for i in D1ranges]

	# hernquist_variable_outer_anis = [williamsevans_selfconsistent_variableouteranisotropyhernquist(1.,i,"hernquist_we/isotropic_centre_D1_"+str(i),"triaxial",spherical_factor=False,descr=r'Williams & Evans, Hernquist, $D_1=$'+str(i)) for i in D1ranges[-1:]]

	D1labels = [r'$D_1=$'+str(i) for i in D1ranges]

	models = hernquist_variable_outer_anis
	ranges = D1ranges
	labels = D1labels

	# D0ranges = np.linspace(0.114,1.814,6)

	# hernquist_variable_inner_anis = [williamsevans_selfconsistent_variableinneranisotropyhernquist(1.,i+0.0001,"hernquist_we/isotropic_farfield_D0_"+str(i)+"_sphr","spherical",spherical_factor=True) for i in D0ranges]

	# hernquist_variable_inner_anis = [williamsevans_selfconsistent_variableinneranisotropyhernquist(1.,i,"hernquist_we/isotropic_farfield_D0_"+str(i),"triaxial",spherical_factor=False,descr=r'Williams & Evans, Hernquist, $D_0=$'+str(i)) for i in D0ranges]

	# D0labels = [r'$D_0=$'+str(i) for i in D0ranges]

	# models = hernquist_variable_inner_anis
	# ranges = D0ranges
	# labels = D0labels

	# b0ranges = np.linspace(0.05,0.48,6)

	# hernquist_variable_inner_anis_method2 = [williamsevans_selfconsistent_variableinneranisotropy_method2_hernquist(1.,i+0.0001,"hernquist_we/isotropic_farfield_b0_"+str(i)+"_sphr","spherical",spherical_factor=True) for i in b0ranges]

	# hernquist_variable_inner_anis_method2 = [williamsevans_selfconsistent_variableinneranisotropy_method2_hernquist(1.,i,"hernquist_we/isotropic_farfield_b0_"+str(i),"triaxial",spherical_factor=False,descr=r'Williams & Evans, Hernquist, $\beta_0=$'+str(i)) for i in b0ranges[:3]]

	# b0labels = [r'$\beta_0=$'+str(i) for i in b0ranges]

	# models = hernquist_variable_inner_anis_method2
	# ranges = b0ranges
	# labels = b0labels

	# qrange = [1.,1.5,1.814]
	# # # b0range=[0.3,0.4,0.5]
	# hernquist_we_models_flat = [williamsevans_selfconsistent(1.,5./3.,5.,q,1.,0.378*np.power((1+1.814)/(1+1.),-5./3.),1.,0.41,0.4,0.,"hernquist_we_t_qz0.4_lmax8_NA4_multiply2_D0centre_"+str(q),ttype='triaxial',descr=r'Williams & Evans, Hernquist, $D_0='+str(q)+r'$, $q_z=0.4$', analytic_model=Hernquist_prof) for q in qrange[:1]]

	# # #+"higherl"

	# # hernquist_we_models_flat = [williamsevans_selfconsistent(1.,5./3.,5.,q,1.,0.378*np.power((1+1.814)/(1+1.),-5./3.),1.,0.41,0.4,0.,"hernquist_we_t_qz0.4_multiply2_D0centre_"+str(q),ttype='triaxial') for q in qrange]

	# models = hernquist_we_models_flat
	# ranges = qrange

	# models=modelsiso[3:]


	# D1ranges=[0.01,0.05,0.114,0.2,0.3,0.4684,0.6456,0.8228,1.0]
	# hernquist_variable_outer_anis = [williamsevans_selfconsistent_variableouteranisotropyhernquist(1.,i,"hernquist_we/isotropic_centre_multiply2_D1_"+str(i),"triaxial",spherical_factor=False,descr=r'Williams & Evans, Hernquist, $D_1=$'+str(i)) for i in D1ranges[1:]]
	# models = hernquist_variable_outer_anis

	# ranges = [0.8]
	# models = [AandCandD_models(A=0.8,C=i,D=0.9,J0=1.,p=10./3.,q=5./3.,s=2.,name="hernquist_ACD_models/A0.8C"+str(i)+"D0.9_hernquist_spherical",ttype="spherical",profile="Hernquist",spherical_factor=2.) for i in ranges]

	# models = [AandCandD_models(A=0.8,C=i,D=0.9,J0=1.,p=10./3.,q=5./3.,s=2.,name="hernquist_ACD_models/A0.8C"+str(i)+"D0.9_hernquist",ttype="triaxial",profile="Hernquist",spherical_factor=1.) for i in ranges]

	def paral_build(x):
		models[x].build()

	def paral_plot(x):
		models[x].plot()

	def paral_veldisp(x):
		models[x].plot_veldisp()

	def paral_splitdensity(x):
		models[x].split_density()

	def paral_projdensity(x):
		models[x].proj_density()

	def paral_highresdensity(x):
		models[x].highres_density()

	def paral_genfunc(x):
		models[x].genfunc()

	def paral_density(x):
		models[x].density()

	pool_size=6
	if(pool_size>len(models)):
		pool_size=len(models)

	if sys.argv[1]=='plot':
		for i in models:
			i.plot(True)

	if sys.argv[1]=='build':
		# p = Pool(pool_size)
		# p.map(paral_build,range(len(models)))
		for i in models:
			i.build()

	if sys.argv[1]=='veldisp':
		p = Pool(pool_size)
		p.map(paral_veldisp,range(len(models)))

	if sys.argv[1]=='split':
		p = Pool(pool_size)
		p.map(paral_splitdensity,range(len(models)))

	if sys.argv[1]=='proj':
		p = Pool(pool_size)
		p.map(paral_projdensity,range(len(models)))

	if sys.argv[1]=='highres':
		p = Pool(pool_size)
		p.map(paral_highresdensity,range(len(models)))

	if sys.argv[1]=='genfunc':
		p = Pool(pool_size)
		p.map(paral_genfunc,range(len(models)))

	if sys.argv[1]=='density':
		p = Pool(pool_size)
		p.map(paral_density,range(len(models)))

	if sys.argv[1]=='veldispstack':
		veldispstack(models,sys.argv[2],labels=labels)

	if sys.argv[1]=='densitycompplot':
		density_comp('June15/hernquist_we_t_qz0.4_D0_1.0.fudgetmp','June15/hernquist_we_t_qz0.4_D0_1.0.genfunc2','hernquist_comp',r'Williams & Evans, Hernquist, $D_0=1$, $q_z=0.4$')
		density_comp('June15/iso_ap1.3_az1.6.fudge','June15/iso_ap1.3_az1.6.genfunc2','iso_comp',r'Binney flattened isochrone, $\alpha_{\phi 0}=1.3,\,\alpha_{z0}=1.6$')
		# density_comp('June15/iso_ap1.3_az1.4.fudge','June15/iso_ap1.3_az1.4.genfunc','iso_ap13_az14_comp',r'Binney flattened isochrone, $\alpha_{\phi 0}=1.3,\,\alpha_{z0}=1.4$')

	# Grid
	# ranges = np.arange(0.05,0.31,0.05)
	# ranges = np.arange(0.7,1.2,0.1)
	# ranges = np.arange(1.2,1.7,0.1)
	# ranges = np.arange(1.7,2.2,0.1)

	# models=[selfconsistent_model([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,10./3.,5./3.,"p3.33q1.66J01_"+str(i)+"_1_1_sphr","spherical","Hernquist") for i in ranges]

	# models=[selfconsistent_model([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,3.,.5,"p3q05J01_"+str(i)+"_1_1","triaxial") for i in ranges]

	# models = [a1a2a3_equal_one_models(ba=1.4,ca=1.6,J0=1.,p=3.,q=0.5,name="p3q05J01_1_1.4_1.6")]
	# models=[selfconsistent_model([2.*np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],2.,3.,0.5,"p3q05J01_"+str(i)+"_1_1_sphr","spherical") for i in ranges]

	# models=[selfconsistent_model([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,3.,.5,"p3q05J01_"+str(i)+"_1_1_axi","axisymmetric") for i in ranges]

	# models=[selfconsistent_model([np.power(i,2./3.),np.power(i,-1./3.),np.power(i,-1./3.)],1.,3.,.5,"p3q05J01_"+str(i)+"_1_1","triaxial") for i in ranges]
	# models = [selfconsistent_model([1.,0.7,1.0],200.,"p3q0J0200_1_0.7_1.0")]

	# models = [selfconsistent_model([sys.argv[2],sys.argv[3],sys.argv[4]],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9])]

