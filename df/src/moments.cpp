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
#include "lmn_orb.h"
#include "df.h"
#include "moments.h"
#include "cuba/cuba.h"
// #include "cubature/cubature.h"
#include <ctime>
#include <string>
#include <thread>

// REMEMBER YOU HAVE CHANGED THE CIRC IN AA.CPP -- FACTOR OF 2

// int density_integrand_cuba_sj(unsigned ndim, long unsigned npts, const double *y, void *fdata, unsigned fdim, double *fval) {
//     density_st *P = (density_st *) fdata;
//     for (unsigned int j = 0; j < npts; ++j) {
//         VecDoub X = {P->x[0],P->x[1],P->x[2],y[j*ndim],y[j*ndim+1],y[j*ndim+2]};
//         fval[j*fdim]=P->DF->realspace_dist(X);
//         fval[j*fdim+1]=X[3]*X[3]*fval[j*fdim];
//         fval[j*fdim+2]=X[4]*X[4]*fval[j*fdim];
//         fval[j*fdim+3]=X[5]*X[5]*fval[j*fdim];
//         fval[j*fdim+4]=X[3]*X[4]*fval[j*fdim];
//         fval[j*fdim+5]=X[3]*X[5]*fval[j*fdim];
//         fval[j*fdim+6]=X[4]*X[5]*fval[j*fdim];
//     }
//     return 0;
// }

// int density_integrand_cuba_split(unsigned ndim, long unsigned npts, const double *y, void *fdata, unsigned fdim, double *fval) {
//     density_st *P = (density_st *) fdata;
//     for (unsigned int j = 0; j < npts; ++j) {
//         VecDoub X = {P->x[0],P->x[1],P->x[2],y[j*ndim],y[j*ndim+1],y[j*ndim+2]};
//         VecDoub G = P->DF->realspace_dist_vec(X);
//         double trigger = 1.;
//         if(P->pp>=0 and G.size()>1){
//             int tt = (int)(G[1]);
//             if((P->pp)!=tt) trigger=0.;
//         }
//         fval[j*fdim]=trigger*G[0];
//     }
//     return 0;
// }


// int density_integrand_cuba_xi(unsigned ndim, long unsigned npts, const double *y, void *fdata, unsigned fdim, double *fval) {
//     density_st *P = (density_st *) fdata;
//     for (unsigned int j = 0; j < npts; ++j) {
//         VecDoub X = {P->x[0],P->x[1],P->x[2],y[j*ndim],y[j*ndim+1],y[j*ndim+2]};
//         fval[j*fdim] =P->DF->realspace_dist(X);
//         fval[j*fdim+1]=X[3]*X[3]*fval[j*fdim];
//         fval[j*fdim+2]=X[P->pp]*X[P->pp]*fval[j*fdim];
//         fval[j*fdim+3]=X[3]*X[P->pp]*fval[j*fdim];
//     }
//     return 0;
// }

int norm_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    density_st *P = (density_st *) fdata; VecDoub X(6,0);
    for(int i=0;i<6;i++) X[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
// Spherical polar
    // double R = sinh(X[0]);
    // fval[0] =R*R*sin(X[2])*sqrt(1+R*R);
    // X[0] = R*cos(X[1])*sin(X[2]);
    // X[1] = R*sin(X[1])*sin(X[2]);
    // X[2] = R*cos(X[2]);
    //fval[0]*=10000.*P->DF->realspace_dist_vec(X)[0]; // density

// Cartesian
    for(int i=0;i<3;i++) X[i]=1./X[i]-1.;
    fval[0] = 10000.*P->DF->realspace_dist_vec(X)[0]; // density
    for(int i=0;i<3;i++) fval[0]*=(1+X[i])*(1+X[i]);

    // if(std::isinf(fval[0])) fval[0]=0.;
    if(std::isinf(fval[0]) or fval[0]!=fval[0] or std::isnan(fval[0])){std::cerr<<"Problem: ";for(auto i:X)std::cout<<i<<" ";std::cout<<fval[0]<<std::endl;}
    return 0;
}

int norm_integrand_cuba_action(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    density_st *P = (density_st *) fdata; VecDoub X(3,0);
    for(int i=0;i<3;i++) X[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    fval[0] = P->DF->action_dist(X);
    return 0;
}


int density_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3];
    density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
    VecDoub G = P->DF->realspace_dist_vec(X); // density
    double trigger = 1.;
    if(P->pp>=0 and G.size()>1){
        int tt = (int)(G[1]);
        if((P->pp)!=tt and !((P->pp)==2 and tt==3)) trigger=0.;
    }
    fval[0]=trigger*G[0];
    return 0;
}

int density_integrand_polar_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3];
    density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    double st = sqrt(1-y2[2]*y2[2]);
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0]*cos(y2[1])*st,y2[0]*sin(y2[1])*st,y2[0]*y2[2]};
    VecDoub G = P->DF->realspace_dist_vec(X); // density
    double trigger = 1.;
    if(P->pp>=0 and G.size()>1){
        int tt = (int)(G[1]);
        if((P->pp)!=tt and !((P->pp)==2 and tt==3)) trigger=0.;
    }
    fval[0]=trigger*G[0]*y2[0]*y2[0];
    return 0;
}

int projected_density_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[4];
    projected_density_st *P = (projected_density_st *) fdata;
    for(int i=0;i<4;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];

    // system coordinates
    VecDoub X = P->phihat*P->x[0]+P->thetahat*P->x[1]+P->n*y2[0];
    for(int i=1;i<4;i++) X.push_back(y2[i]);

    VecDoub G = P->DF->realspace_dist_vec(X); // density
    double trigger = 1.;
    if(P->pp>=0 and G.size()>1){
        int tt = (int)(G[1]);
        if((P->pp)!=tt and !((P->pp)==2 and tt==3)) trigger=0.;
    }
    fval[0]=trigger*G[0];
    return 0;
}

int sigmaxx_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3]; density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
    fval[0]=X[3]*X[3]*P->DF->realspace_dist(X);
    return 0;
}

int sigmayy_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3]; density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
    fval[0]=X[4]*X[4]*P->DF->realspace_dist(X);
    return 0;
}

int sigmazz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3]; density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
    fval[0]=X[5]*X[5]*P->DF->realspace_dist(X);
    return 0;
}

int sigmaxy_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3]; density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
    fval[0]=X[3]*X[4]*P->DF->realspace_dist(X);
    return 0;
}

int sigmaxz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3]; density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
    fval[0]=X[3]*X[5]*P->DF->realspace_dist(X);
    return 0;
}

int sigmayz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3]; density_st *P = (density_st *) fdata;
    for(int i=0;i<3;i++) y2[i]=(P->x2max[i]-P->x2min[i])*y[i]+P->x2min[i];
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
    fval[0]=X[4]*X[5]*P->DF->realspace_dist(X);
    return 0;
}

int veldist_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata){
    double y2[3];   veldist_st *P = (veldist_st *) fdata;
    int count = 0;
    for(int i=0;i<3;i++){
        if(i==P->swit) y2[i]=P->x[3];
        else{
            y2[i]=(P->x2max[count]-P->x2min[count])*y[count]+P->x2min[count];
            count++;
        }
    }
    VecDoub X = {P->x[0],P->x[1],P->x[2],y2[0],y2[1],y2[2]};
    fval[0] = P->DF->realspace_dist(X);
    // printVector(X);std::cout<<fval[0]<<std::endl;
    return 0;
}

const int nproc = 1;
const int SEED = time(0);

double integrate(integrand_t integrand, density_st *P, double IE, double AE, std::string type, double *err){

    int neval,fail,nregions;
    double integral[1],error[1],prob[1];
    int NSIZE = P->x2min.size();
    double prod = 1.;
    for(int i=0;i<NSIZE;i++)prod*=(P->x2max[i]-P->x2min[i]);

    if(type=="Vegas")
        Vegas(NSIZE,nproc,integrand,P,1,IE,AE,0,SEED,
        MINEVAL,MAXEVAL,NSTART,NINCREASE,NBATCH,GRIDNO,STATEFILE,SPIN,
        &neval,&fail,integral,error,prob);

    else if (type=="Suave")
        Suave(NSIZE,nproc,integrand,P,1,IE,AE,0,SEED,
        MINEVAL,MAXEVAL,NNEW,FLATNESS,STATEFILE,SPIN,&nregions,
        &neval,&fail,integral,error,prob);

    else if (type=="Cuhre")
        Cuhre(NSIZE,nproc,integrand,P,1,IE,AE,0,
        MINEVAL, MAXEVAL, 0, STATEFILE,SPIN,
        &nregions, &neval, &fail, integral, error, prob);

    else
        Divonne(NSIZE,nproc,integrand,P,1,IE,AE,0,SEED,
        MINEVAL, MAXEVAL, KEY1, KEY2, KEY3, MAXPASS,
        BORDER, MAXCHISQ, MINDEVIATION,
        NGIVEN, LDXGIVEN, nullptr, NEXTRA, nullptr,STATEFILE,SPIN,
        &nregions, &neval, &fail, integral, error, prob);
    if(err)*err=prod*error[0];
    if(fail!=0)std::cerr<<"Error: Required accuracy not reached." <<std::endl;
    return prod*integral[0];
}
