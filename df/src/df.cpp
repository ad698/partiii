//============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "GSLInterface/GSLInterface.h"
#include "gnuplot/gnuplot_i.h"
#include <gsl/gsl_poly.h>
#ifdef TORUS
#include "falPot.h"
#endif
#include "utils.h"
#include "coordsys.h"
#include "coordtransforms.h"
#include "potential.h"
#include "orbit.h"
#include "stackel_aa.h"
#include "spherical_aa.h"
#include "lmn_orb.h"
#include "df.h"
#include "cuba/cuba.h"
// #include "cubature/cubature.h"
#include <ctime>
#include <string>
#include <thread>
#include "moments.h"

//============================================================================

double df::normalization(double IE, std::string type){

	double Ve = -2.*pot->Phi({1e-9,1e-9,1e-9})+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::normalization\n";
	Ve=sqrt(Ve);

	double lowerlim = 1./1.0000001, upperlim = 0.01;
	VecDoub x2min = {5e-4,5e-4,5e-4,-Ve,-Ve,0.};
	VecDoub x2max = {lowerlim,lowerlim,lowerlim,Ve,Ve,Ve};
	density_st P(this,{0.},x2min,x2max,-1);
	// double D = 0.;
	double err;
	double D = 16.*integrate(&norm_integrand_cuba,&P,IE,0,"Divonne",&err)/10000.;
	print(err);
	print(D);
	// // for(int i=1;i<subs.size();i++){
	// // IE = 1e-3;
	x2min[0]=upperlim;x2max[0]=lowerlim;x2min[1]=lowerlim;x2max[1]=1.;x2min[2]=lowerlim;x2max[2]=1.;
	P = density_st(this,{0.},x2min,x2max,-1);
	D+=16.*integrate(&norm_integrand_cuba,&P,IE,1,"Divonne")/10000.;print(D);

	x2min[0]=lowerlim;x2max[0]=1.;x2min[1]=upperlim;x2max[1]=lowerlim;x2min[2]=lowerlim;x2max[2]=1.;
	P = density_st(this,{0.},x2min,x2max,-1);
	D+=16.*integrate(&norm_integrand_cuba,&P,IE,1,"Divonne")/10000.;print(D);

	x2min[0]=lowerlim;x2max[0]=1.;x2min[1]=lowerlim;x2max[1]=1.;x2min[2]=upperlim;x2max[2]=lowerlim;
	P = density_st(this,{0.},x2min,x2max,-1);
	D+=16.*integrate(&norm_integrand_cuba,&P,IE,1,"Divonne")/10000.;print(D);

	x2min[0]=upperlim;x2max[0]=lowerlim;x2min[1]=upperlim;x2max[1]=lowerlim;x2min[2]=lowerlim;x2max[2]=1.;
	P = density_st(this,{0.},x2min,x2max,-1);
	D+=16.*integrate(&norm_integrand_cuba,&P,IE,1,"Divonne")/10000.;print(D);

	x2min[0]=upperlim;x2max[0]=lowerlim;x2min[1]=lowerlim;x2max[1]=1.;x2min[2]=0.;x2max[2]=lowerlim;
	P = density_st(this,{0.},x2min,x2max,-1);
	D+=16.*integrate(&norm_integrand_cuba,&P,IE,1,"Divonne")/10000.;print(D);

	x2min[0]=lowerlim;x2max[0]=1.;x2min[1]=upperlim;x2max[1]=lowerlim;x2min[2]=upperlim;x2max[2]=lowerlim;
	P = density_st(this,{0.},x2min,x2max,-1);
	D+=16.*integrate(&norm_integrand_cuba,&P,IE,1,"Divonne")/10000.;print(D);

	return D;
}

double df::normalization_action(double IE, std::string type){
	// The loop orbits can rotate in two senses so we must
	// multiply the integral by two for these orbits
	// However, we use Jl twice the expected value for the loop orbits
	// This factor cancels with the factor of 2 from the clockwise
	// and anti-clockwise rotating loops
	// Therefore, the integral is simply from 0 to inf for all three actions

	VecDoub x2min = {0.0,0.0,0.0};
	VecDoub x2max = {10000.,10000.,10000.};
	density_st P(this,{0.},x2min,x2max,-1);
	double D = pow(2.*PI,3)*integrate(&norm_integrand_cuba_action,&P,IE,0,type);
	return D;
}

double df::density(const VecDoub& x, double IE, std::string type){
	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::density\n";
	Ve=sqrt(Ve);
	VecDoub x2min = {-Ve+0.00001,-Ve+0.00001,-Ve+0.00001};
	VecDoub x2max = {Ve,Ve,Ve};
	density_st P(this,x,x2min,x2max,-1);
	double D = integrate(&density_integrand_cuba,&P,IE,0,type);
	return D;
}

double df::density_pol(const VecDoub& x, double IE, std::string type){
	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::density\n";
	Ve=sqrt(Ve);
	VecDoub x2min = {0.,0.,-1.};
	VecDoub x2max = {Ve,2.*PI,1.};
	density_st P(this,x,x2min,x2max,-1);
	double D = integrate(&density_integrand_polar_cuba,&P,IE,0,type);
	return D;
}


double df::projected_density(const VecDoub& x, const VecDoub& phi_theta, double size_in_zdirection, double IE, std::string type){
	double cp = cos(phi_theta[0]),sp=sin(phi_theta[0]);
	double ct = cos(phi_theta[1]),st=sin(phi_theta[1]);
	VecDoub phihat={-sp,cp,0.}, thetahat={-cp*ct,-ct*sp,st};
	VecDoub Xmin = phihat*x[0]+thetahat*x[1];

	double Ve = -2.*pot->Phi(Xmin)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::projected_density\n";
	Ve=sqrt(Ve);
	VecDoub x2min = {-size_in_zdirection,-Ve+SMALL,-Ve+SMALL,-Ve+SMALL};
	VecDoub x2max = {size_in_zdirection,Ve,Ve,Ve};
	projected_density_st P(this,x,phi_theta,x2min,x2max,-1);
	double D = integrate(&projected_density_integrand_cuba,&P,IE,0,type);
	return D;
}

void df::oneD_density(const std::string& ofile, double xmin, double xmax, int n, std::string type, double IE){
	VecDoub range;
	if(type=="log")
		range = create_log_range<double>(xmin,xmax,n);
	else
		range = create_range<double>(xmin,xmax,n);

	oneD_density(ofile,range,IE);

	return;
}

void df::oneD_density(const std::string& ofile, VecDoub range, double IE){

	std::ofstream out; out.open(ofile);
	out<<"# x rho_x rho_y rho_z rho_xyz\n";
	#pragma omp parallel for schedule(dynamic)
	for(unsigned i=0;i<range.size();++i){
		// double rho1 = density({range[i],1e-3,1e-3},IE);
		// std::cerr<<"Done"<<std::endl;
		// double rho2 = density({1e-3,range[i],1e-3},IE);
		// double rho3 = density({1e-3,1e-3,range[i]},IE);
		double r = range[i]/sqrt(3.);
		double rho4 = density({r,r,r},IE);
		// out<<range[i]<<" "<<rho1<<" "<<rho2<<" "<<rho3<<" "<<rho4<<std::endl;
		out<<range[i]<<" "<<rho4<<std::endl;
	}
	out.close();
	return;
}

VecDoub df::split_density(const VecDoub& x, double IE){

	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::split_density\n";
	Ve=sqrt(Ve);

	VecDoub x2min = {-Ve+TINY,-Ve+TINY,-Ve+TINY};
	VecDoub x2max = {Ve,Ve,Ve};
	density_st P(this,x,x2min,x2max,0);
	VecDoub Results;
	Results.push_back(integrate(&density_integrand_cuba,&P,IE,0));
	std::cout<<P.pp<<", ";
	P.pp  = 1;
	Results.push_back(integrate(&density_integrand_cuba,&P,IE,0));
	std::cout<<P.pp<<", ";
	P.pp  = 2;
	Results.push_back(integrate(&density_integrand_cuba,&P,IE,0));
	std::cout<<P.pp<<std::endl;
	P.pp  = 3;
	Results.push_back(integrate(&density_integrand_cuba,&P,IE,0));
	return Results;
}
void df::projected_density(const std::string& ofile, double xmax, int n, double IE){

	VecDoub r_range = create_log_range<double>(0.01,xmax,n);
	VecDoub theta_range = create_range<double>(0.,2.*PI-0.1,n/2);

	std::ofstream out; out.open(ofile);
	out<<"# x y rho\n";
	#pragma omp parallel for schedule(dynamic)
	for(unsigned i=0;i<theta_range.size();++i){
	for(auto j:r_range){
		double x = j*cos(theta_range[i]), y = j*sin(theta_range[i]);
		double rho = projected_density({x,y},{PI/4.,PI/4.},20.,IE);
		out<<x<<" "<<y<<" "<<rho<<std::endl;
	}}
	out.close();
	return;
}
void df::density(const std::string& ofile, double xmin, double xmax, int n, std::string type , double IE){

	VecDoub range;
	if(type=="log")
		range = create_log_range<double>(xmin,xmax,n);
	else
		range = create_range<double>(xmin,xmax,n);

	std::ofstream out; out.open(ofile);
	out<<"# x y/z rho_xy rho_xz\n";
	#pragma omp parallel for schedule(dynamic)
	for(unsigned i=0;i<range.size();++i){
		for(unsigned j=0;j<range.size();++j){
			double rho1 = density({range[i],range[j],1e-3},IE);
			double rho2 = density({range[i],1e-3,range[j]},IE);
			out<<range[i]<<" "<<range[j]<<" "<<rho1<<" "<<rho2<<std::endl;
		}
	}
	out.close();
	return;
}

void df::split_density(const std::string& ofile, double xmin, double xmax, int n, std::string type , double IE){

	VecDoub range;
	if(type=="log")
		range = create_log_range<double>(xmin,xmax,n);
	else
		range = create_range<double>(xmin,xmax,n);

	std::ofstream out; out.open(ofile);
	out<<"# x box sal ilal olal\n";
	// std::vector<unsigned> V = {18,20};
	// for(auto i: V){
	#pragma omp parallel for schedule(dynamic)
	for(unsigned i=0;i<range.size();++i){
		VecDoub xy = split_density({range[i],range[i],range[i]},IE);
		out<<range[i]<<" "<<xy[0]<<" "<<xy[1]<<" "<<xy[2]<<" "<<xy[3]<<std::endl;
	}
	out.close();
	return;
}

double df::veldist(const VecDoub& x, int comp, double IE){

	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::veldist\n";
	Ve=sqrt(Ve);

	VecDoub x2min = {-Ve+0.00001,-Ve+0.00001};
	VecDoub x2max = {Ve,Ve};
	veldist_st P(this,x,x2min,x2max,comp);
	return integrate(&veldist_integrand_cuba,&P,IE,0);
}

// double df::density_cub(const VecDoub& x, double IE, int dd){

// 	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
// 	if(Ve<0.) std::cerr<<"Negative velocity in df::density_cub\n";
// 	Ve=sqrt(Ve);

// 	VecDoub x2min = {-Ve,-Ve,-Ve};
// 	VecDoub x2max = {Ve,Ve,Ve};
// 	density_st P(this,x,x2min,x2max,dd);

// 	int FSIZE = 1;
// 	double integral[FSIZE],error[FSIZE];
// 	int NSIZE = P.x2min.size();
// 	double xm[3]={P.x2min[0],P.x2min[1],P.x2min[2]};
// 	double xp[3]={P.x2max[0],P.x2max[1],P.x2max[2]};

// 	hcubature_v(FSIZE, density_integrand_cuba_split, &P,
//                 NSIZE, xm, xp,
//                 MAXEVAL, EPSABS, IE,
//                 ERROR_INDIVIDUAL, integral, error);

// 	return integral[0];
// }

// VecDoub df::moments(const VecDoub& x, double IE){

// 	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
// 	if(Ve<0.) std::cerr<<"Negative velocity in df::moments\n";
// 	Ve=sqrt(Ve);
// 	VecDoub x2min = {-Ve,-Ve,-Ve};
// 	VecDoub x2max = {Ve,Ve,Ve};
// 	density_st P(this,x,x2min,x2max,-1);

// 	int FSIZE = 7;
// 	double integral[FSIZE],error[FSIZE];
// 	int NSIZE = P.x2min.size();
// 	double xm[3]={P.x2min[0],P.x2min[1],P.x2min[2]};
// 	double xp[3]={P.x2max[0],P.x2max[1],P.x2max[2]};

// 	hcubature_v(FSIZE, density_integrand_cuba_sj, &P,
//                 NSIZE, xm, xp,
//                 MAXEVAL/10, EPSABS, IE,
//                 ERROR_INDIVIDUAL, integral, error);

// 	VecDoub integralV(FSIZE,0);
// 	for(int i=0;i<FSIZE;i++)integralV[i]=integral[i];
// 	return integralV;
// }

// VecDoub df::split_density2(const VecDoub& X, double IE){
// 	VecDoub R (4,0);
// 	for(int i=0;i<4;i++){
// 		if(i==3) R[i]=density_cub(X, IE, -1);
// 		else	 R[i]=density_cub(X, IE, i);
// 		std::cout<<R[i]<<std::endl;
// 	}
// 	return R;
// }

VecDoub df::testJeans(const VecDoub& x){
	// finds the components of the triaxial Jean's equation
	// LHS[i] = \rho \frac{\partial\Phi/\partial x_i}
	// RHS[i] = \frac{\partial\sigma^2_{ij}}{\partial x_j}

	double Delta = 0.005;
	VecDoub xtmp = x;
	VecDoub potDerivs = PotGrad(x);
	VecDoub central = moments(x);
	std::cerr<<"Central found"<<std::endl;

	VecDoub LHS = {potDerivs[0]*central[0],potDerivs[1]*central[0],potDerivs[2]*central[0]};
	VecDoub xplus(7,0), xminus(7,0), yplus(7,0), yminus(7,0), zplus(7,0), zminus(7,0);

	xtmp[0]+=Delta;
	xplus = moments(xtmp);
	xtmp[0]-=2.*Delta;
	xminus = moments(xtmp);
	std::cerr<<"x found"<<std::endl;
	xtmp[0]=x[0];

	xtmp[1]+=Delta;
	yplus = moments(xtmp);
	xtmp[1]-=2.*Delta;
	yminus = moments(xtmp);
	std::cerr<<"y found"<<std::endl;
	xtmp[1]=x[1];

	xtmp[2]+=Delta;
	zplus = moments(xtmp);
	xtmp[2]-=2.*Delta;
	zminus = moments(xtmp);
	std::cerr<<"z found"<<std::endl;

	VecDoub RHS(3,0);
	RHS[0] -= (xplus[1]-xminus[1])/Delta/2.;
	RHS[0] -= (yplus[4]-yminus[4])/Delta/2.;
	RHS[0] -= (zplus[5]-zminus[5])/Delta/2.;

	RHS[1] -= (xplus[4]-xminus[4])/Delta/2.;
	RHS[1] -= (yplus[2]-yminus[2])/Delta/2.;
	RHS[1] -= (zplus[6]-zminus[6])/Delta/2.;

	RHS[2] -= (xplus[5]-xminus[5])/Delta/2.;
	RHS[2] -= (yplus[6]-yminus[6])/Delta/2.;
	RHS[2] -= (zplus[3]-zminus[3])/Delta/2.;

	return {central[0],central[1],central[2],central[3],central[4],central[5],central[6],LHS[0],LHS[1],LHS[2],RHS[0],RHS[1],RHS[2]};
}

void df::twoDdensity(const std::string& ofile){
	// std::ofstream out; out.open(ofile);
	#pragma omp parallel for schedule(dynamic)
	for(int i=0;i<40;i++){
		double x = (double)i*0.25+0.01;
		// df DF(*this);
		// std::cout<<density({x,1.,0.01})<<std::endl;
		for(double z = 0.01;z<10.;z+=0.25)
			std::cout<<x<<" "<<z<<" "<<density({x,z,0.01},1e-3)<<" "<<density({x,0.01,z},1e-3)<<std::endl;
	}
	// out.close();
	return;
}

void df::velocity_distributions(const VecDoub& x, const std::string& ofile, double IE){
	std::ofstream out; out.open(ofile);
	VecDoub rr = x; rr.push_back(0.);
	for(double v = -150.;v<150.;v+=5.){
		rr[3]=v;
		out<<v<<" "<<veldist(rr, 0, IE)<<" "<<veldist(rr, 1, IE)<<" "<<veldist(rr, 2 ,IE)<<std::endl;
	}
	out.close();
	return;
}

// void df::low_action_test(void){
// 	VecDoub X = {-0.406569,0.682708,5.44297e-06,66.6536,13.6172,5.85178e-06};
// 	printVector(ActionCalculator->find_actions(X));
// 	return;
// }

VecDoub df::moments(const VecDoub& x, double IE){

	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::moments\n";
	Ve=sqrt(Ve);

	VecDoub x2min = {-Ve,-Ve,-Ve};
	VecDoub x2max = {Ve,Ve,Ve};
	density_st P(this,x,x2min,x2max,-1);
	// double IE = 1e-4;
	VecDoub moms = {integrate(&density_integrand_cuba,&P,IE,1e-11),
					integrate(&sigmaxx_integrand_cuba,&P,IE,1e-9),
					integrate(&sigmayy_integrand_cuba,&P,IE,1e-9),
					integrate(&sigmazz_integrand_cuba,&P,IE,1e-9),
					integrate(&sigmaxy_integrand_cuba,&P,IE,1e-9),
					integrate(&sigmaxz_integrand_cuba,&P,IE,1e-9),
					integrate(&sigmayz_integrand_cuba,&P,IE,1e-9)};
	return moms;
}

VecDoub df::spherical_moments(double r){
	// i gives the component to find the vel disps sigma_{xi}
	// Using the P struct pp to store i

	VecDoub x = {r,0.001,0.001};

	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::xymoments\n";
	Ve=sqrt(Ve);

	VecDoub x2min = {-Ve,-Ve,-Ve};
	VecDoub x2max = {Ve,Ve,Ve};
	density_st P(this,x,x2min,x2max,-1);
	VecDoub moms = {integrate(&density_integrand_cuba,&P,1e-3,0),
					integrate(&sigmaxx_integrand_cuba,&P,1e-3,0),
					integrate(&sigmayy_integrand_cuba,&P,1e-3,0)};
	return moms;
}

VecDoub df::xymoments(const VecDoub& x,double IE,int with_cross){
	// i gives the component to find the vel disps sigma_{xi}
	// Using the P struct pp to store i

	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::xymoments\n";
	Ve=sqrt(Ve);

	VecDoub x2min = {-Ve,-Ve,-Ve};
	VecDoub x2max = {Ve,Ve,Ve};
	density_st P(this,x,x2min,x2max,-1);
	VecDoub moms = {integrate(&density_integrand_cuba,&P,IE,0),
					integrate(&sigmaxx_integrand_cuba,&P,IE,0),
					integrate(&sigmayy_integrand_cuba,&P,IE,0)};
	if(with_cross) moms.push_back(integrate(&sigmaxy_integrand_cuba,&P,IE,0));
	return moms;
}

VecDoub df::xzmoments(const VecDoub& x,double IE,int with_cross){
	// i gives the component to find the vel disps sigma_{xi}
	// Using the P struct pp to store i

	double Ve = -2.*pot->Phi(x)+2.*pot->Phi_max();
	if(Ve<0.) std::cerr<<"Negative velocity in df::xzmoments\n";
	Ve=sqrt(Ve);

	VecDoub x2min = {-Ve,-Ve,-Ve};
	VecDoub x2max = {Ve,Ve,Ve};
	density_st P(this,x,x2min,x2max,-1);
	VecDoub moms = {integrate(&density_integrand_cuba,&P,IE,0),
					integrate(&sigmaxx_integrand_cuba,&P,IE,0),
					integrate(&sigmazz_integrand_cuba,&P,IE,0)};

	if(with_cross) moms.push_back(integrate(&sigmaxz_integrand_cuba,&P,IE,0));
	return moms;
}

void df::xidisp(const std::string& ofile, double xmin, double xmax, int n, std::string type , double IE,int spherical){

	VecDoub range;
	if(type=="log")
		range = create_log_range<double>(xmin,xmax,n);
	else
		range = create_range<double>(xmin,xmax,n);

	std::ofstream out; out.open(ofile);
	if(spherical){
		out<<"# r rho sigmarr sigmapp\n";
		for(auto z: range){
			VecDoub xy = xymoments({z,0.001,0.001},IE,0);
			out<<z<<" "<<xy[0]<<" "<<xy[1]<<" "<<xy[2]<<std::endl;
		}
	}else{
		out<<"# x y/z rho sigmaxx sigmayy sigmaxy rho sigmaxx sigmazz sigmaxz\n";
		#pragma omp parallel for schedule(dynamic)
		for(unsigned i=0;i<range.size();++i)
			for(auto z: range){
				VecDoub xy = xymoments({range[i],z,0.005},IE);
				VecDoub xz = xzmoments({range[i],0.005,z},IE);
				out<<range[i]<<" "<<z<<" "<<xy[0]<<" "<<xy[1]<<" "<<xy[2]<<" "<<xy[3]<<" "<<xz[0]<<" "<<xz[1]<<" "<<xz[2]<<" "<<xz[3]<<std::endl;
			}
		}
	out.close();
	return;
}

void df::xidisp(const std::string& ofile, double xmin, double xmax, double dx,double IE,int spherical){
	int n = (int)((xmax-xmax)/dx);
	xidisp(ofile,xmin,xmax,n,"linear",IE,spherical);
	return;
}

void df::sphr_disp(const std::string& ofile, double xmin, double xmax, double dx){
	int n = (int)((xmax-xmax)/dx);
	xidisp(ofile,xmin,xmax,n,"linear",5e-3,1);
	return;
}

int numericalnorm_integrand_cuba(const int ndim[],const double s[], const int*fdim, double fval[], void *fdata){
    VecDoub actions(3,0.); double prod=1.;
    density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++){
    	if(s[i]==1. or s[i]==0.){fval[0]=0.; return 0;}
    	prod*=(1.-s[i])*(1.-s[i]);
    	actions[i]=P->scale*s[i]/(1.-s[i]);
    }
    fval[0]=P->DF->action_dist(actions)/prod;
    return 0;
}

double df::numerical_norm(double scale){
	// Integrates over action space
	// Note this is a factor of two out for the spherical case as it
	// doesn't count the two senses of rotation for the loop orbits
	// For the triaxial case it works because of cancellation between
	// the definition of the radial action for the loop orbits and
	// the two senses of rotation for the loops
	VecDoub x2min(3,0.), x2max(3,1.);
	density_st P(this,VecDoub(3,0.),x2min,x2max,-1, scale);
	double fac=2.; if(symmetry=="triaxial") fac=1.;
	return fac*pow(2.*PI*scale,3)*integrate(&numericalnorm_integrand_cuba,&P,1e-3,0,"Divonne");
}

//============================================================================
// Single power law df
//============================================================================

double df_single_power::action_dist(const VecDoub& J){
	double JJ = 0.;
	for(int i=0;i<3;i++) JJ+=fabs(J[i])*a_i[i];
	double P = pow(10./(JJ),p);
	if(P!=P or std::isinf(P))P=0.;
	return Mass*P;
}

//============================================================================
// Double power law df
//============================================================================

double df_double_power::action_dist(const VecDoub& J){
	double Jn = sqrt(J[0]*J[0]+J[1]*J[1]+J[2]*J[2]);
	double JJ =fabs(Jfac*J[0])* /*a_i[i];//*/(a_i[0]+Jn/J0)/(1.+Jn/J0);
	for(int i=1;i<3;i++)
		JJ+=fabs(J[i])* /*a_i[i];//*/(a_i[i]+Jn/J0)/(1.+Jn/J0);
	JJ = pow(JJ,s);
	double P = pow(JJ+pow(J0,s),-p/s);
	if(q!=0) P*=pow(JJ+pow(Jc,s),-q/s);
	if(P!=P or std::isinf(P)) P=0.;
	return MN*P;
}

double df_double_power::realspace_dist(const VecDoub& X){

	if(pot->H(X)>0.)return 0.;
	VecDoub J;
	if(use_fixed_ab){
		J = ActionCalculator->actions(X,&ab[0]);
	}
	else{
		J = ActionCalculator->actions(X);
	}
	return action_dist(J);
}

VecDoub df_double_power::realspace_dist_vec(const VecDoub& X){
	if(pot->H(X)>0.){return {0.,0.};}
	VecDoub J;
	if(use_fixed_ab){
		Actions_TriaxialStackel_Fudge ATSF(pot,-30.,-20.);
		VecDoub TrueJ = ATSF.actions(X);
		J = ActionCalculator->actions(X,&ab[0]);
		for(int i=0;i<3;i++)av[i]+=pow(J[i]-TrueJ[i],2.);
		double aJ = action_dist(J);
		av[3]+=aJ*(pow(J[0]-TrueJ[0],2.)+pow(J[1]-TrueJ[1],2.)+pow(J[2]-TrueJ[2],2.));
		av[4]+=aJ;
	}
	else
		J = ActionCalculator->actions(X);
	if(noise>0)
		for(int i=0;i<3;i++) J[i]=exp(log(fabs(J[i]))+rnGauss->nextnumber()*noise/fabs(J[i]));
	if(printing){
	for(auto i:X)std::cout<<i<<" ";
	for(auto i:J)std::cout<<i<<" ";
	std::cout<<action_dist(J)<<std::endl;
	}
	return {action_dist(J),J[3]};
}

//============================================================================
// Posti Binney df
//============================================================================

double df_pb::action_dist(const VecDoub& J){
	double h = fabs(Jfac*J[0])+.5*(fabs(J[1])+fabs(J[2]));
	double g = fabs(Jfac*J[0])+fabs(J[1])+fabs(J[2]);
	double P = pow(1+J0/h,(6-alpha)/(4-alpha));
	P*=pow(1+g/J0,3-2.*beta);
	if(P!=P or std::isinf(P)) P=0.;
	return MN*P;
}

//==========================================================z==================
// Williams Evans df
//============================================================================

double df_we::action_dist(const VecDoub& J){
	double L = fabs(J[1])+fabs(J[2]);
	double absJ = sqrt(Jfac*Jfac*J[0]*J[0]+L*L);
	L = fabs(J[1])+fabs(J[2])/qz;
	double D = (D0+D1*absJ/Jb)/(1.+absJ/Jb);
	double T = (Sa+Sg*absJ/J0)/(1.+absJ/J0);
	double curlyL = L+D*fabs(Jfac*J[0]);
	double P = pow((J0*J0+curlyL*curlyL),-0.5*(mu-lambda));
	P*=T*pow(curlyL,-lambda);
	if(b0!=0.) P*=pow(1+Jb/L,2.*b0);
	if(P!=P or std::isinf(P)) P=0.;
	return MN*P;
}

//============================================================================
// Deformed isochrone df
//============================================================================

double df_isochrone::action_dist(const VecDoub& J){

	VecDoub coeff_i(3,0);
	coeff_i[1] = a_i[0];
	coeff_i[2] = a_i[1];

	double L = fabs(J[1])+fabs(J[2]);
	double E = fabs(J[0])*Jfac+.5*(L+sqrt(L*L+4*GM*b));
	E = GM/E;
	E *= E; // This is -2H

	double Jbar = 2.*GM/sqrt(E)-sqrt(GM*GM/(E)+3.*GM*b);
	Jbar/=3.;
	double psi = tanh(Jbar/sqrt(GM*b));
	double OmegaR = E*sqrt(E)/GM;
	double OmegaL = 0.5*OmegaR*(1+Jbar/sqrt(Jbar*Jbar+GM*b));
	double alphar0=1-OmegaL/OmegaR*(coeff_i[1]+coeff_i[2]-2.);
	double alpha0 = 1-OmegaL/(OmegaR+OmegaL)*(coeff_i[2]-1.);

	coeff_i[0] = (1.-psi)*alpha0+psi*alphar0;
	coeff_i[1] = (1-psi)*alpha0+psi*a_i[1];

	if(coeff_i[0]<0. or coeff_i[1]<0.)
		std::cerr<<"One of the coefficients is less than zero: alpha_r = "<<coeff_i[0]<<" ,alpha_phi = "<<coeff_i[1]<<" "<<alpha0<<" "<<alphar0<<" "<<Jbar<<" "<<psi<<" "<<OmegaL/OmegaR<<std::endl;

	double JR = Jfac*coeff_i[0]*fabs(J[0]);
	L = fabs(J[1])*coeff_i[1]+fabs(J[2])*coeff_i[2];
	double H = GM*b*0.5/pow(JR+0.5*(L+sqrt(L*L+4.*GM*b)),2.);
	double mH = 1.-H, H2 = H*H;
	double f = 27.-66.*H+320.*H2-240.*H2*H+64.*H2*H2+3.*(16.*H2+28.*H-9.)*asin(sqrt(H))/sqrt(H*mH);
	f*=sqrt(H)/pow(2*mH,4.)/sqrt(2.)/pow(2.*PI,3.)/pow(GM*b,1.5);
	for(int i=0;i<3;i++) f*=coeff_i[i];
	if(f!=f) f=0.;
	return MN*f;
}

//============================================================================
// Spherical isochrone df
//============================================================================

double df_isochrone_spherical::action_dist(const VecDoub& J){
	double JR = fabs(J[0]);
	double L = fabs(J[1])+fabs(J[2]);
	double H = GM*b*0.5/pow(JR+0.5*(L+sqrt(L*L+4.*GM*b)),2.);
	double mH = 1.-H, H2 = H*H;
	double f = 27.-66.*H+320.*H2-240.*H2*H+64.*H2*H2+3.*(16.*H2+28.*H-9.)*asin(sqrt(H))/sqrt(H*mH);
	f*=sqrt(H)/pow(2*mH,4.)/sqrt(2.)/pow(2.*PI,3.)/pow(GM*b,1.5);
	return MN*f;
}

double df_isochrone_spherical::realspace_dist(const VecDoub& X){
	if(pot->H(X)>0.)return 0.;
	Actions_Spherical AS(dynamic_cast<SphericalPotential*>(pot));
	VecDoub J = AS.actions(X);
	return action_dist(J);
}

VecDoub df_isochrone_spherical::realspace_dist_vec(const VecDoub& X){
	if(pot->H(X)>0.){return {0.,0.};}
	Actions_Spherical AS(dynamic_cast<SphericalPotential*>(pot));
	VecDoub J = AS.actions(X);
	return {action_dist(J),J[3]};
}

//============================================================================
