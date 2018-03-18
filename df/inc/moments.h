#ifndef MOMENTS_H
#define MOMENTS_H

#include "cuba/cuba.h"
#include "df.h"

extern const int SEED;

// int density_integrand_cuba_sj(unsigned ndim, long unsigned npts, const double *y, void *fdata, unsigned fdim, double *fval);
// int density_integrand_cuba_split(unsigned ndim, long unsigned npts, const double *y, void *fdata, unsigned fdim, double *fval);
int norm_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int norm_integrand_cuba_action(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int density_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int density_integrand_polar_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int projected_density_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int sigmaxx_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int sigmayy_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int sigmazz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int sigmaxy_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int sigmaxz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int sigmayz_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
int veldist_integrand_cuba(const int ndim[],const double y[], const int*fdim, double fval[], void *fdata);
double integrate(integrand_t integrand, density_st  *P, double IE, double AE, std::string type = "Divonne",double *err=nullptr);
#endif
