#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "GSLInterface/GSLInterface.h"
#include "gnuplot/gnuplot_i.h"
#include <gsl/gsl_poly.h>
#include "falPot.h"
#include "utils.h"
#include "coordsys.h"
#include "coordtransforms.h"
#include "potential.h"
#include "orbit.h"
#include "stackel_aa.h"
#include "genfunc_aa.h"
#include "lmn_orb.h"
#include "df.h"
#include "cuba/cuba.h"
#include "cubature/cubature.h"
#include <ctime>
#include <string>
#include <thread>
#include "moments.h"

static const int num_threads = 4;

void velocity_plots(double a){
	df_double_power DF({1.,a,1.88},10.,3.,0.);
	DF.velocity_distributions({4.,0.5,0.5},std::to_string(a)+"_velocity.dat",1e-3);
	return;
}

void density_plots(double a){
	df_double_power DF({1.,a,1.88},10.,3.,0.);
	DF.twoDdensity(std::to_string(a)+"_2D_density.dat");
	return;
}

void quicktest(double IE){
	// Test integration speeds
	// For IE = 1e-4
	// Vegas 1149.16 101.702
	// Suave 1082.26 99.8139
	// Cuhre 1147.89 30.3759
	// Divonne 1147.94 22.6485
	// Cubature 1147.94 194.145
	df_double_power DF({1.,1.,1.88},10.,3.,0.);
	std::vector<std::string> types = {"Vegas", "Suave", "Cuhre", "Divonne"};
	for(auto i: types){
		std::cout<<i<<" ";
		auto A = std::chrono::high_resolution_clock::now();
		double D = DF.density({1.,1.,1.},IE,i);
		auto B = std::chrono::high_resolution_clock::now();
		std::cout<<D<<" "<<std::chrono::duration_cast<std::chrono::nanoseconds>(B-A).count()/1e9<<std::endl;
	}
	std::cout<<"Cubature ";
	auto A = std::chrono::high_resolution_clock::now();
	double D = DF.density_cub({1.,1.,1.},IE,-1);
	auto B = std::chrono::high_resolution_clock::now();
	std::cout<<D<<" "<<std::chrono::duration_cast<std::chrono::nanoseconds>(B-A).count()/1e9<<std::endl;
	return;
}

void test_moments_stackel(void){
	// add artificial noise
	std::ofstream outfile; outfile.open("density_noise.dat2");
	StackelTriaxial SF(3.61/500.,-30.,-20.);
	df_double_power DF({1.,0.7,1.88},10.,3.,0.,&SF,nullptr,{-30,-20.});
	std::vector<VecDoub> X = {{4.,0.5,0.5},{0.5,4.,0.5},{0.5,0.5,4.},{4.,4.,4.}};
	outfile<<0.<<" ";
	for(auto i:X)outfile<<DF.density(i,1e-4)<<" ";
	outfile<<std::endl;
	for(double I = 0.1;I<105.;I*=2.){
		DF.set_noise(I);
		outfile<<I<<" ";
		for(auto i:X)outfile<<DF.density(i,1e-3)<<" ";
		outfile<<std::endl;
	}
	outfile.close();
}

void test_moments_stackel2(void){
	// change alpha, beta
	std::ofstream outfile; outfile.open("density_beta.dat2");
	StackelTriaxial SF(3.61/500.,-30.,-20.);
	df_double_power DF({1.,0.7,1.88},10.,3.,0.,&SF,nullptr,{-30,-20.});
	std::vector<VecDoub> X = {{4.,0.5,0.5},{0.5,4.,0.5},{0.5,0.5,4.},{4.,4.,4.}};
	outfile<<-20.<<" ";
	for(auto i:X)outfile<<DF.density(i,1e-4)<<" ";
	for(int i=0;i<16;i++)outfile<<"0. ";
	outfile<<std::endl;
	for(double b = -19.5;b<-5.;b+=0.75){
		DF.set_ab({-30.,b});
		outfile<<b<<" ";
		for(auto i:X){
			outfile<<DF.density(i,1e-3)<<" ";
			for(int j=0;j<3;j++) outfile<<sqrt(DF.av[j]/(double)DF.evals)<<" ";
			outfile<<sqrt(DF.av[3]/(double)DF.evals/3./DF.av[4])<<" ";
			DF.evals=0; DF.av = VecDoub(5,0);
		}
		outfile<<std::endl;
	}
	outfile.close();
}

void test_norm_stackel(void){
	std::ofstream outfile; outfile.open("norm_alpha.dat");
	StackelTriaxial SF(3.61/500.,-30.,-20.);
	df_double_power DF({1.,0.7,1.88},10.,4.5,0.,&SF,nullptr,{-30,-20.});
	double TrueNorm = DF.analytic_norm();
	std::cout<<TrueNorm<<" "<<DF.normalization(1e-4)<<std::endl;
	for(double a = -29.5;a<20.5;a+=1.){
		DF.set_ab({a,-20.});
		outfile<<a<<" "<<DF.normalization(1e-4)-TrueNorm<<" ";
		for(int j=0;j<3;j++) outfile<<sqrt(DF.av[j]/(double)DF.evals)<<" ";
		outfile<<sqrt(DF.av[3]/(double)DF.evals/3./DF.av[4])<<std::endl;
		DF.evals=0; DF.av = VecDoub(5,0);
	}
	outfile.close();
}

void xidisp_plots(double a){
	df_double_power DF({1.,a,1.88},10.,3.,0.);
	DF.xidisp(std::to_string(a)+"_xi_density_tmp.dat");
	return;
}


void fullJeanstest(int i, double a){
	df_double_power DF({1.,a,1.88},10.,3.,0.);
	std::ofstream outfile;
	outfile.open("jeans_a"+std::to_string(a)+std::to_string(i)+".dat");
	VecDoub X = {1.,1.,1.};
	double down = log(0.5), up = log(30.);
	int NNMAX = 30;
	for(int nn=0;nn<NNMAX;nn++){
		double x = exp(down+(up-down)*(double)nn/(double)(NNMAX-1));
		if(i<3)X[i]=x;
		else{ X[0]=x; X[1]=0.5*x; X[2]=x/3.; }
		outfile<<x<<" ";for(auto j: DF.testJeans(X)) outfile<<j<<" "; outfile<<std::endl;
	}
	outfile.close();
}

void multiJeanstest(double a){
	std::thread t[num_threads];
	for (int i = 0; i < num_threads; ++i) {
        t[i] = std::thread(fullJeanstest,i,a);
    }
    for (int i = 0; i < num_threads; ++i) {
    	t[i].join();
    }
    return;
}

void fulldensitytest(int i,double a){
	df_double_power DF({1.,a,1.88},10.,3.,0.);
	std::cout<<"Launched "<<i<<std::endl;
	std::ofstream outfile;
	outfile.open("a"+std::to_string(a)+"_density_split.dat");
	VecDoub X = {1.,1.,1.};
	double down = log(0.5), up = log(30.);
	int NNMAX = 30;
	for(int nn=0;nn<NNMAX;nn++){
		double x = exp(down+(up-down)*(double)nn/(double)(NNMAX-1));
		if(i<3)	X[i]=x;
		else{ X[0]=x; X[1]=0.5*x; X[2]=x/3.; }
		outfile<<x<<" ";for(auto j: DF.split_density(X,1e-3))  outfile<<j<<" ";
		outfile<<std::endl;
	}
	outfile.close();
}

void multidensitytest(double a){
	std::thread t[num_threads];
	for (int i = 0; i < num_threads; ++i) {
        t[i] = std::thread(fulldensitytest,i,a);
    }
    for (int i = 0; i < num_threads; ++i) {
    	t[i].join();
    }
    return;
}

void test_isochrone_df(void){

	// NFW Pot(10.,12.3e5,0.95,0.85);
	Isochrone Pot(1.,4.,0.98,0.96);
	df_isochrone DF({1.,1.,1.},1.,4.,&Pot);

    int NMAX = 4;
    VecDoub xx(NMAX,0), exact(NMAX,0);

    #pragma omp parallel for schedule(dynamic)
    for(int xn = 0; xn<NMAX; xn++){
        double x = 0.1*(double)xn+.1;
        xx[xn] = x;
        exact[xn] = DF.density({x,0.5,0.5});
    }

    Gnuplot G("points ls 1");
    G.set_xrange(0.9*Min(xx),1.1*Max(xx));
    G.set_yrange(0.9*Min(exact),1.1*Max(exact));
    G.set_xlabel("x").set_ylabel("Density");
    G.savetotex("isochrone_density_test").plot_xy(xx,exact);
    G.outputpdf("isochrone_density_test");
}

void comparing_genfunc_and_fudge_density(double a){
	std::ofstream outfile;
	outfile.open("genfunc_fudge_comparison_"+std::to_string(a)+".dat");
	NFW Pot(10.,12.3e5,0.95,0.85);
	Actions_Genfunc AG(&Pot);
	df_double_power DF_G({1.,a,1.88},10.,3.,0.,&Pot,&AG);
	df_double_power DF_F({1.,a,1.88},10.,3.,0.,&Pot);
	double x;
	for(double lx = -1.; lx < 2.; lx+=0.2){
		x = pow(10.,lx);
		outfile<<x<<" ";
		outfile<<DF_G.density({x,0.05,0.05},1e-3)<<" ";
		outfile<<DF_F.density({x,0.05,0.05},1e-3)<<std::endl;
	}
	outfile<<"# "<<DF_G.evals<<" "<<DF_F.evals<<std::endl;
	outfile.close();
}


int main(){
	// comparing_genfunc_and_fudge_density(0.7);
	// comparing_genfunc_and_fudge_density(3.28);
	// return 0;


	// fulldensitytest(0,0.7);
	// multiJeanstest(0.7);
	// return 0;
	// Logarithmic Pot(220.,0.95,0.85);
	// StackelTriaxial Pot(3.61/500.,-30.,-20.);

	// test_isochrone_df();
	// return 0;

	// Actions_TriaxialStackel_Fudge ATSF(&Pot,-156.3405,-49.6996);
	// printVector(ATSF.actions({0.748294,1.09128e-08,18.9486,-60.7449,-56.8977,3.72989 }));
	// return 0;
	// df DF({1.,1.,1.},10.,2.,1.,&Pot);
	// printVector(DF.lmn->find_actions({5.,5.,5.,0.5,0.5,0.5}));
	// DF.twoDdensity("spherical_model_J0-10_p-2_q-1.density");
	// return 0;

	// xidisp_plots(0.7);
	// xidisp_plots(3.28);
	// df_double_power DF({1.,3.28,1.88},10.,3.5,0.);
	// printVector(DF.moments({4.,0.5,0.001},1e-2));

	// Isochrone Pot(2e6,6.,0.995,0.99);
	NFW Pot(10.,12.3e5,0.95,0.85);
	// StackelTriaxial Pot(3.61/500.,-30.,-20.);
	Actions_TriaxialStackel_Fudge AG(&Pot,-30.,-20.);
	df_isochrone DF({1.,1.,1.},2e6,6.,&Pot,&AG,0);
    // std::cout<<SEED<<std::endl;
    print(DF.density({5.,5.,5.},1e-2,"Divonne"));

    print(DF.density({1.,1.,1.},1e-2,"Divonne"));
    print(DF.density({1.,1.,.1},1e-2,"Divonne"));
    print(DF.density({2.,5.,.1},1e-2,"Divonne"));
	// print(DF.normalization(1e-2));
}
	// 2.63546e12
	// return 0;
	// printVector(DF.moments({1.,1.,1.},1e-2));

	// quicktest(1e-4);
	// velocity_plots(3.28);
	// velocity_plots(0.7);
	// density_plots(3.28);
	// density_plots(0.7);
	// fulldensitytest(0,3.28);
	// fulldensitytest(0,0.7);
	// multiJeanstest(0.7);
	// multiJeanstest(3.28);

	// for(int i=0;i<4;i++)fullJeanstest(i, 0.7);

	// test_moments_stackel2();

	// test_norm_stackel();







	// multidensitytest();
	// df DF(3.28,1.88,10.,3.);
	// printVector(DF.testJeans({3.,1.,1.},4));
	// return 0;
	// for(double x = 0.5;x<10.;x+=0.5)std::cout<<x<<" "<<DF.density({x,0.1,0.1})<<std::endl;

	// printVector(fit_NFW_density(&DF));
	// density_from_potential(&DF);
	// return 0;
	// for(double x = 1.;x<1.5;x+=0.5){
	// 	VecDoub X = {5.,x,3.};
	// 	std::cout<<x<<" "<<std::time(NULL)<<std::endl;printVector(DF.moments(X));std::cout<<std::time(NULL)<<std::endl;
	// 	printVector(DF.moments_mc(X));std::cout<<std::time(NULL)<<std::endl;
	// }
	// std::thread t[num_threads];
	// for (int i = 0; i < num_threads; ++i) {
	//  	VecDoub yy = {3.,3.,1.*i+0.01};
 //        t[i] = std::thread(moments,yy);
 //    }
 //    for (int i = 0; i < num_threads; ++i) {
 //    	t[i].join();
 //    }
	// for(double x = 5.; x<6.0001;x+=0.05){
	// 	/*std::cout<<x<<" ";*/printVector(DF.moments({x,3.,1.}));
	// 	// DF.testJeans({x,3.0,3.0});
	// }
// }

//
// int density_integrand_cuba_sj_separate(unsigned ndim, long unsigned npts, const double *y, void *fdata, unsigned fdim, double *fval) {
// 	density_st *P = (density_st *) fdata;
// 	for (unsigned int j = 0; j < npts; ++j) {
// 		VecDoub X = {P->x[0],P->x[1],P->x[2],y[j*ndim],y[j*ndim+1],y[j*ndim+2]};
// 		int p = 0;
// 		fval[j*fdim]  = P->DF->realspace_dist(X);
// 		fval[j*fdim+1]= fval[j*fdim];
// 		fval[j*fdim+2]= fval[j*fdim];
// 		if(p<0.5){ fval[j*fdim+1]=0.; fval[j*fdim+2]=0.;}
// 		else if(p>0.5 and p<1.5){ fval[j*fdim]=0.; fval[j*fdim+2]=0.;}
// 		else if(p>1.5 and p<2.5){ fval[j*fdim]=0.; fval[j*fdim+1]=0.;}
// 	}
// 	return 0;
// }

// static int density_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
// 	double y2[3];	density_st *P = (density_st *) fdata;
// 	for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
// 	VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
// 	VecDoub G = P->DF->realspace_dist_vec(X); // density
// 	double trigger = 1.;
// 	if(P->pp>=0 and G.size()>1){
// 		int tt = (int)(G[1]);
// 		if((P->pp)!=tt) trigger=0.;
// 	}
// 	fval[0]=trigger*G[0];
// 	return 0;
// }

// static int sigmaxx_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
// 	double y2[3]; density_st *P = (density_st *) fdata;
// 	for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
// 	VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
// 	fval[0]=X[3]*X[3]*P->DF->realspace_dist(X);
// 	return 0;
// }

// static int sigmayy_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
// 	double y2[3]; density_st *P = (density_st *) fdata;
// 	for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
// 	VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
// 	fval[0]=X[4]*X[4]*P->DF->realspace_dist(X);
// 	return 0;
// }

// static int sigmazz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
// 	double y2[3]; density_st *P = (density_st *) fdata;
// 	for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
// 	VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
// 	fval[0]=X[5]*X[5]*P->DF->realspace_dist(X);
// 	return 0;
// }

// static int sigmaxy_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
// 	double y2[3]; density_st *P = (density_st *) fdata;
// 	for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
// 	VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
// 	fval[0]=X[3]*X[4]*P->DF->realspace_dist(X);
// 	return 0;
// }

// static int sigmaxz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
// 	double y2[3]; density_st *P = (density_st *) fdata;
// 	for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
// 	VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
// 	fval[0]=X[3]*X[5]*P->DF->realspace_dist(X);
// 	return 0;
// }

// static int sigmayz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
// 	double y2[3]; density_st *P = (density_st *) fdata;
// 	for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
// 	VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
// 	fval[0]=X[4]*X[5]*P->DF->realspace_dist(X);
// 	return 0;
// }

//

// double integrate(integrand_t integrand, density_st *P, double IE, double AE){

// 	int neval,fail,nregions;
// 	double integral[1],error[1],prob[1];
// 	int NSIZE = P->x2min.size();
// 	double prod = 1.;
// 	for(int i=0;i<NSIZE;i++)prod*=(P->x2max[i]-P->x2min[i]);
// 	// prod = 1.;
	// Vegas(NSIZE,1,integrand,P,IE,EPSABS,0,1,
	// MINEVAL,MAXEVAL,NSTART,NINCREASE,NBATCH,GRIDNO,STATEFILE,
	// &neval,&fail,integral,error,prob);

// 	// Suave(3,1,density_integrand_cuba,&P,IE,EPSABS,0,1,
// 	// MINEVAL,MAXEVAL,NNEW,FLATNESS,STATEFILE,&nregions,
// 	// &neval,&fail,integral,error,prob);

//   	// Cuhre(NSIZE,1,integrand,P,IE,EPSABS,0,
//    //  MINEVAL, MAXEVAL, 0, STATEFILE,
//    //  &nregions, &neval, &fail, integral, error, prob);

//  	Divonne(NSIZE,1,integrand,P,IE,AE,0,1,
//  	  MINEVAL, MAXEVAL, KEY1, KEY2, KEY3, MAXPASS,
//  	  BORDER, MAXCHISQ, MINDEVIATION,
//  	  NGIVEN, LDXGIVEN, NULL, NEXTRA, NULL,STATEFILE,
//  	  &nregions, &neval, &fail, integral, error, prob);

// 	return prod*integral[0];
// }

// double df::density(VecDoub x, double IE){
// 	double Ve=sqrt(-2.*(nfw->Phi(x)/*-nfw->Phi({100.,100.,100.})*/));
// 	VecDoub x2min = {-Ve+0.00001,-Ve+0.00001,-Ve+0.00001};
// 	VecDoub x2max = {Ve,Ve,Ve};
// 	density_st P(this,x,x2min,x2max,-1);
// 	double D = integrate(&density_integrand_cuba,&P,IE,0);
// 	return D;
// }

// VecDoub df::split_density(VecDoub x, double IE){
// 	double Ve=sqrt(-2.*nfw->Phi(x));
// 	VecDoub x2min = {-Ve+0.00001,-Ve+0.00001,-Ve+0.00001};
// 	VecDoub x2max = {Ve,Ve,Ve};
// 	int dd = 0;
// 	density_st P(this,x,x2min,x2max,dd);
// 	double f = integrate(&density_integrand_cuba,&P,IE,0);
// 	VecDoub Results = {f};
// 	std::cout<<P.pp<<", ";
// 	P.pp  = 1;
// 	f = integrate(&density_integrand_cuba,&P,IE,0);
// 	Results.push_back(f);
// 	std::cout<<P.pp<<", ";
// 	P.pp  = 2;
// 	f = integrate(&density_integrand_cuba,&P,IE,0);
// 	Results.push_back(f);
// 	std::cout<<P.pp<<std::endl;
// 	return Results;
// }

// VecDoub df::moments_mc(VecDoub x){

// 	double Ve=sqrt(-2.*(nfw->Phi(x)/*-nfw->Phi({0.,100.,0.})*/));
// 	VecDoub x2min = {-Ve,-Ve,-Ve};
// 	VecDoub x2max = {Ve,Ve,Ve};
// 	density_st P(this,x,x2min,x2max,-1);
// 	double IE = 1e-4;
// 	VecDoub moms = {integrate(&density_integrand_cuba,&P,IE,1e-11),
// 					integrate(&sigmaxx_integrand_cuba,&P,IE,1e-9),
// 					integrate(&sigmayy_integrand_cuba,&P,IE,1e-9),
// 					integrate(&sigmazz_integrand_cuba,&P,IE,1e-9),
// 					integrate(&sigmaxy_integrand_cuba,&P,IE,1e-9),
// 					integrate(&sigmaxz_integrand_cuba,&P,IE,1e-9),
// 					integrate(&sigmayz_integrand_cuba,&P,IE,1e-9)};
// 	return moms;
// }


// double nfw_true_density(VecDoub x,double norm, double r0, double a, double b){
// 	double m = sqrt(x[0]*x[0]+x[1]*x[1]/a/a+x[2]*x[2]/b/b);
// 	return norm/m/(1+m/r0)/(1+m/r0);
// }

// double fitNFW(const gsl_vector *v, void *params){
// 	df *DF = (df *) params;
// 	if(gsl_vector_get(v,0)<0. or gsl_vector_get(v,1)<0. or gsl_vector_get(v,2)>1. or gsl_vector_get(v,3)>1.) return 1e10;
// 	double sum = 0.;
// 	for (double x=0.5;x<50.;x+=2.)
// 		for (double y=0.5;y<50.;y+=2.)
// 			for (double z=0.5;z<50.;z+=2.){
// 				sum+=pow(log(DF->nfw->density({x,y,z}))
// 					-log(nfw_true_density({x,y,z},gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2),gsl_vector_get(v,3))),2.);
// 			}
// 	std::cout<<sum<<std::endl;
// 	return sum;
// }

// VecDoub fit_NFW_density(df *DF){
// 	VecDoub a1 = {1./120.,10.,0.95,0.85};
// 	VecDoub sizes ={0.01,1.,0.01,0.01};
// 	minimiser min(&fitNFW,a1,sizes,1e-3,DF);
// 	VecDoub results;
// 	min.minimise(&results,10000,0);
// 	return results;
// }

// void density_from_potential(df *DF){
// 	// Long-axis
// 	for (double x=10.5;x<50.;x+=2.)
// 		for (double y=10.5;y<50.;y+=2.)
// 			for (double z=10.5;z<50.;z+=2.)
// 	{
// 		std::cout<<x<<" "<<y<<" "<<z<<" "<<nfw_true_density({x,y,z},121.521,53.2441,1,0.660122)<<" "<<DF->nfw->density({x,y,z})<<std::endl;//" "<<DF->density({x,0.1,0.1})<<std::endl;
// 	}
// 	return;
// }
