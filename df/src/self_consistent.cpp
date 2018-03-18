// ============================================================================
// ----------------------
// Self-consistent models
// ----------------------
//
// Allows construction of spherical, axisymmetric and triaxial action-based
// self-consistent equilibria.
//
// Multiple components are supported (each with their own separate multipole
// grid), and one can add an external potential (e.g. central black hole)
//
// ============================================================================

#include <iostream>
#include <vector>
#include <memory>
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
#include "spherical_aa.h"
#include "stackel_aa.h"
#include "lmn_orb.h"
#include "cuba/cuba.h"
// #include "cubature/cubature.h"
#include <ctime>
#include <string>
#include <thread>
#include "df.h"
#include "moments.h"
#include "Multipole.h"
#include "self_consistent.h"

// ============================================================================

const std::string data_folder = "data/";

void selfcon::output_pot(VecDoub radial_grid,Potential_JS *InitPot,std::string name,std::string type){
    std::ofstream outfile; outfile.open(name);
    for(auto i: radial_grid)
        if(type=="spherical" or "axisymmetric")
            for(auto j: create_range(0.,PI/2.,8)){
            if(type=="triaxial")
                for(auto k: create_range(0.,PI/2.,8))
                    outfile<<i<<" "<<
                    InitPot->Phi(conv::SphericalPolarToCartesian({i,j,k}))
                    <<std::endl;
            else
                outfile<<i<<" "<<
                InitPot->Phi(conv::SphericalPolarToCartesian({i,0.,j}))
                <<std::endl;
            }
        else
            outfile<<i<<" "<<
                InitPot->Phi(conv::SphericalPolarToCartesian({i,0.,0.}))
                <<std::endl;
    outfile.close();
}

// ============================================================================

void SphericalSelfConsistent::make_self_consistent(unsigned NITER, double gamma)
{
    // Now iterate
    unsigned n=0;
    IncrementalSphericalPotential *Pot_grid[NITER+1];
    Pot_grid[0] = new IncrementalSphericalPotential(InitPot,ME,gamma,ExternalPot);
    MultipoleExpansion_Spherical *ME_grid[NITER];
    bool are_we_there_yet=false;
    while(n<NITER and !are_we_there_yet){
        DF->reset_pot(Pot_grid[n]);
        ME_grid[n] = new MultipoleExpansion_Spherical(&DDD,ME->nradial(),a0,innerradius,outerradius);
        outerradius=ME_grid[n]->outerradius();
        ME_grid[n]->visualize(name+"_it"+std::to_string(n)+".vis");
        ME_grid[n]->output_to_file(name+".me");
        selfcon::printMmaxtolog(name,n+1,ME_grid[n]->Mass(outerradius));

        are_we_there_yet=true;
        for(auto i:ME_grid[n]->get_radial_grid()){
            double P = ME_grid[n]->Phi({i,0.,0.});
            if(ExternalPot)
                P+=ExternalPot->Phi({i,0.,0.});
            if(fabs((Pot_grid[n]->Phi({i,0.,0.})-P)/P)>threshold)
                are_we_there_yet=false;
        }

        Pot_grid[n+1]=new IncrementalSphericalPotential(Pot_grid[n],ME_grid[n],gamma,ExternalPot);
        n++;
        if(are_we_there_yet)
            break;
    }
    if(are_we_there_yet) selfcon::stamplogfile(name,n);
    // Copy updated potential back to DF and clean up
    DF->reset_pot(std::move(new MultipoleExpansion_SphericalPotential(name+".me")));
    pot=DF->pot;
    for(unsigned i=0;i<n+1;++i) delete Pot_grid[i];
    for(unsigned i=0;i<n;++i) delete ME_grid[i];
}

void SphericalMultiComSelfConsistent::make_self_consistent(unsigned NITER, double gamma)
{
    // Now iterate
    unsigned n=0;
    IncrementalSphericalPotential *Pot_grid[NITER+1];
    Pot_grid[0] = new IncrementalSphericalPotential(InitPot,TotalPot,gamma,ExternalPot);
    std::vector<std::vector<MultipoleExpansion_Spherical*>> ME_grid;
    std::vector<MultiComponentPotential<MultipoleExpansion_Spherical>*> TotalPotGrid = {TotalPot};

    bool are_we_there_yet=false;
    while(n<NITER){
        DFs->reset_pot(Pot_grid[n]);
        std::vector<MultipoleExpansion_Spherical*> MEgt;
        for(unsigned i=0;i<DFs->ncompts();++i){
            MEgt.push_back(new MultipoleExpansion_Spherical(&(DDD[i]),ME[i]->nradial(),a0[i],innerradius[i],outerradius[i]));
            fill_radial_grid();
            MEgt[i]->visualize(name+"_it"+std::to_string(n)+".vis"+std::to_string(i));
            MEgt[i]->output_to_file(name+".me"+std::to_string(i));
        }
        ME_grid.push_back(MEgt);
        TotalPotGrid.push_back(new MultiComponentPotential<MultipoleExpansion_Spherical>(ME_grid[n]));

        are_we_there_yet=true;
        for(auto i:radial_grid){
            double P = TotalPotGrid[n+1]->Phi({i,0.,0.});
            if(ExternalPot)
                P+=ExternalPot->Phi({i,0.,0.});
            if(fabs((TotalPotGrid[n]->Phi({i,0.,0.})-P)/P)>threshold)
                are_we_there_yet=false;
        }

        Pot_grid[n+1]=new IncrementalSphericalPotential(Pot_grid[n],TotalPotGrid[n],gamma,ExternalPot);
        n++;
        if(are_we_there_yet)
            break;
    }
    if(are_we_there_yet) selfcon::stamplogfile(name,n);
    DFs->reset_pot(std::move(TotalPotGrid[n-1]));pot=DFs->pot;
    for(unsigned i=0;i<n;++i) delete Pot_grid[i];
    for(unsigned i=0;i<ME_grid.size();++i)
        for(unsigned j=0;j<ME_grid[0].size();++j) delete ME_grid[i][j];
    for(unsigned i=0;i<TotalPotGrid.size();++i) delete TotalPotGrid[i];
}

// ============================================================================

void AxisymmetricSelfConsistent::make_self_consistent(unsigned NITER, double gamma)
{
    // Now iterate

    IncrementalPotential *Pot_grid[NITER+1];
    Pot_grid[0] = new IncrementalPotential(InitPot,ME,gamma,ExternalPot);
    MultipoleExpansion_Axisymmetric *ME_grid[NITER];

    unsigned n=0;
    bool are_we_there_yet=false;
    while(n<NITER){

        DF->reset_pot(Pot_grid[n]);
        ME_grid[n] = new MultipoleExpansion_Axisymmetric(&DDD,ME->nradial(),ME->nangular(),ME->lmax(),a0,innerradius,outerradius);
        outerradius=ME_grid[n]->outerradius();
        ME_grid[n]->visualize(name+"_it"+std::to_string(n)+".vis");
        ME_grid[n]->output_to_file(name+".me");
        selfcon::printMmaxtolog(name,n+1,ME_grid[n]->Mass(outerradius));

        are_we_there_yet=true;
        for(auto i:ME_grid[n]->get_radial_grid())
            for(auto j: create_range(0.,PIH,8)){
                VecDoub Pol = {i*sin(j),0.,j*cos(j)};
                double P = ME_grid[n]->Phi(Pol);
                if(ExternalPot)
                    P+=ExternalPot->Phi(Pol);
                if(fabs((Pot_grid[n]->Phi(Pol)-P)/P)>threshold)
                    are_we_there_yet=false;
            }

        Pot_grid[n+1]=new IncrementalPotential(Pot_grid[n],
                                               ME_grid[n],gamma,ExternalPot);
        n++;
        if(are_we_there_yet)
            break;
    }
    if(are_we_there_yet) selfcon::stamplogfile(name,n);
    DF->reset_pot(std::move(new MultipoleExpansion_Axisymmetric(name+".me")));
    pot=DF->pot;
    for(unsigned i=0;i<n+1;++i) delete Pot_grid[i];
    for(unsigned i=0;i<n;++i) delete ME_grid[i];
}

// ============================================================================

void AxisymmetricMultiComSelfConsistent::make_self_consistent(unsigned NITER, double gamma)
{
    // Now iterate
    unsigned n=0;
    // Series of linear combos of potentials
    IncrementalPotential *Pot_grid[NITER+1];
    Pot_grid[0] = new IncrementalPotential(InitPot,TotalPot,gamma,ExternalPot);
    // Grid (each step) of grid (each DF) of multipole expansions
    std::vector<std::vector<MultipoleExpansion_Axisymmetric*>> ME_grid;
    std::vector<MultiComponentPotential<MultipoleExpansion_Axisymmetric>*> TotalPotGrid = {TotalPot};

    bool are_we_there_yet=false;
    while(n<NITER){
        DFs->reset_pot(Pot_grid[n]);
        std::vector<MultipoleExpansion_Axisymmetric*> MEgt;
        for(unsigned i=0;i<DFs->ncompts();++i){
            MEgt.push_back(new MultipoleExpansion_Axisymmetric(&(DDD[i]),ME[i]->nradial(),ME[i]->nangular(),ME[i]->lmax(),a0[i],innerradius[i],outerradius[i]));
            fill_radial_grid();
            MEgt[i]->visualize(name+"_it"+std::to_string(n)+".vis"+std::to_string(i));
            MEgt[i]->output_to_file(name+".me"+std::to_string(i));
        }
        ME_grid.push_back(MEgt);
        TotalPotGrid.push_back(new MultiComponentPotential<MultipoleExpansion_Axisymmetric>(ME_grid[n]));

        are_we_there_yet=true;
        // use one of the grids --> could be improved
        for(auto i:radial_grid)
            for(auto j: create_range(0.,PIH,8)){
                VecDoub Pol = {i*sin(j),0.,j*cos(j)};
                double P = TotalPotGrid[n+1]->Phi(Pol);
                if(ExternalPot)
                    P+=ExternalPot->Phi(Pol);
                if(fabs((TotalPotGrid[n]->Phi(Pol)-P)/P)>threshold)
                    are_we_there_yet=false;
            }

        Pot_grid[n+1]=new IncrementalPotential(Pot_grid[n],TotalPotGrid[n],gamma,ExternalPot);
        n++;
        if(are_we_there_yet)
            break;
    }
    if(are_we_there_yet) selfcon::stamplogfile(name,n);
    DFs->reset_pot(std::move(TotalPotGrid[n-1]));pot=DFs->pot;
    for(unsigned i=0;i<n;++i) delete Pot_grid[i];
    for(unsigned i=0;i<ME_grid.size();++i)
        for(unsigned j=0;j<ME_grid[0].size();++j) delete ME_grid[i][j];
    for(unsigned i=0;i<TotalPotGrid.size();++i) delete TotalPotGrid[i];
}


// ============================================================================

void TriaxialSelfConsistent::make_self_consistent(unsigned NITER, double gamma)
{
    // Now iterate

    IncrementalPotential *Pot_grid[NITER+1];
    Pot_grid[0] = new IncrementalPotential(InitPot,ME,gamma,ExternalPot);
    MultipoleExpansion_Triaxial *ME_grid[NITER];

    unsigned n=0;
    bool are_we_there_yet=false;
    while(n<NITER){

        DF->reset_pot(Pot_grid[n]);
        ME_grid[n] = new MultipoleExpansion_Triaxial(&DDD,ME->nradial(),ME->nangular(),ME->lmax(),ME->mmax(),a0,innerradius,outerradius);
        outerradius=ME_grid[n]->outerradius();
        ME_grid[n]->visualize(name+"_it"+std::to_string(n)+".vis");
        ME_grid[n]->output_to_file(name+".me");
        selfcon::printMmaxtolog(name,n+1,ME_grid[n]->Mass(outerradius));

        are_we_there_yet=true;
        for(auto i:ME_grid[n]->get_radial_grid())
            for(auto j: create_range(0.,PIH,8))
                for(auto k: create_range(0.,PIH,8)){
                    VecDoub X = {i*cos(j)*sin(k),i*sin(j)*sin(k),i*cos(k)};
                    double P = ME_grid[n]->Phi(X);
                    if(ExternalPot)
                        P+=ExternalPot->Phi(X);
                    if(fabs((Pot_grid[n]->Phi(X)-P)/P)>threshold)
                        are_we_there_yet=false;
                }

        Pot_grid[n+1]=new IncrementalPotential(Pot_grid[n],ME_grid[n],gamma,ExternalPot);
        n++;
        if(are_we_there_yet)
            break;
    }
    if(are_we_there_yet) selfcon::stamplogfile(name,n);
    DF->reset_pot(std::move(new MultipoleExpansion_Triaxial(name+".me")));
    pot=DF->pot;
    for(unsigned i=0;i<n+1;++i) delete Pot_grid[i];
    for(unsigned i=0;i<n;++i) delete ME_grid[i];
}

VecDoub TriaxialSelfConsistent::find_triaxiality(double r){
    double zero = 1e-3;
    double dens_x = DDD.density({r,zero,zero});
    return {DDD.density({zero,r,zero})/dens_x,
            DDD.density({zero,zero,r})/dens_x};
}
void TriaxialSelfConsistent::plot_triaxiality(void){
    VecDoub x = create_range(1e-3,20.,10), f1, f2;
    for(auto i:x){
        VecDoub t = find_triaxiality(i);
        f1.emplace_back(t[0]);
        f2.emplace_back(t[1]);
    }
    Gnuplot G("lines ls 1");
    G.set_xrange(0.9*Min(x),1.1*Max(x));
    G.set_yrange(0.9*Min(f2),1.1*Max(f1));
    G.set_xlabel("x").set_ylabel("Triaxiality");
    G.savetotex("Triaxiality").plot_xy(x,f1);
    G.set_style("lines ls 2").plot_xy(x,f2);
    G.outputpdf("Triaxiality");

}

// ============================================================================

void TriaxialMultiComSelfConsistent::make_self_consistent(unsigned NITER, double gamma)
{
    // Now iterate
    unsigned n=0;
    // Series of linear combos of potentials
    IncrementalPotential *Pot_grid[NITER+1];
    Pot_grid[0] = new IncrementalPotential(InitPot,TotalPot,gamma,ExternalPot);
    // Grid (each step) of grid (each DF) of multipole expansions
    std::vector<std::vector<MultipoleExpansion_Triaxial*>> ME_grid;
    std::vector<MultiComponentPotential<MultipoleExpansion_Triaxial>*> TotalPotGrid;

    bool are_we_there_yet=false;
    while(n<NITER){
        DFs->reset_pot(Pot_grid[n]);
        std::vector<MultipoleExpansion_Triaxial*> MEgt;
        for(unsigned i=0;i<DFs->ncompts();++i){
            MEgt.push_back(new MultipoleExpansion_Triaxial(&(DDD[i]),ME[i]->nradial(),ME[i]->nangular(),ME[i]->lmax(),ME[i]->mmax(),a0[i],innerradius[i],outerradius[i]));
            fill_radial_grid();
            MEgt[i]->visualize(name+"_it"+std::to_string(n)+".vis"+std::to_string(i));
            MEgt[i]->output_to_file(name+".me"+std::to_string(i));
        }
        ME_grid.push_back(MEgt);
        TotalPotGrid.push_back(new MultiComponentPotential<MultipoleExpansion_Triaxial>(ME_grid[n]));

        are_we_there_yet=true;
        for(auto i:radial_grid)
            for(auto j: create_range(0.,PIH,8))
                for(auto k: create_range(0.,PIH,8)){
                    VecDoub X = {i*cos(j)*sin(k),i*sin(j)*sin(k),i*cos(k)};
                    double P = TotalPotGrid[n+1]->Phi(X);
                    if(ExternalPot)
                        P+=ExternalPot->Phi(X);
                    if(fabs((TotalPotGrid[n]->Phi(X)-P)/P)>threshold)
                        are_we_there_yet=false;
                }


        Pot_grid[n+1]=new IncrementalPotential(Pot_grid[n],TotalPotGrid[n],gamma,ExternalPot);
        n++;
        if(are_we_there_yet)
            break;
    }
    if(are_we_there_yet) selfcon::stamplogfile(name,n);
    DFs->reset_pot(std::move(TotalPotGrid[n-1]));pot=DFs->pot;
    for(unsigned i=0;i<n;++i) delete Pot_grid[i];
    for(unsigned i=0;i<ME_grid.size();++i)
        for(unsigned j=0;j<ME_grid[0].size();++j) delete ME_grid[i][j];
    for(unsigned i=0;i<TotalPotGrid.size();++i) delete TotalPotGrid[i];
}

// ============================================================================


 //    PowerLaw Pot(2e6,1.,.9,.8);
    // lmn_orb LMN(&Pot,1.,40.);
 //    df_single_power DFI({1.,1.,1.},2e6,3.,&Pot,&LMN);
 //    TriaxialSelfConsistent TSC(&DFI,&Pot,"test",1.,0.01,10.);
 //    TSC.make_self_consistent(10,0.);
 //    return 0;


    // TSC5.xidisp(data_folder+std::string(argv[5])+".xdisp",0.1,100.,8.);

	// df_double_power DF({1.,1.,1.},100.,3.01,0.,&Pot,{},1.5e7/conv::G);
 //    TriaxialSelfConsistent TSC(&DF,&Pot,"nfw_start/1_1_1_v2");
 //    TSC.make_self_consistent(5,0.);

    // lmn_orb LMN(&Pot,0.01,500.);
    // df_double_power DF2({1.,1.,1.3},100.,3.01,0.,&Pot,&LMN,{},1.5e7/conv::G);
    // TriaxialSelfConsistent TSC2(&DF2,&Pot,"nfw_start/200215_1_1_13",1.,0.01,100.,20,6,6);
    // TSC2.make_self_consistent(5,0.);

    // lmn_orb LMN(&Pot,0.5,100.);
    // df_double_power DF3({1.,1.3,1.},100.,3.01,0.,&Pot,&LMN,{},1.5e7/conv::G);
    // TriaxialSelfConsistent TSC3(&DF3,&Pot,"nfw_start/200215_1_13_1");
    // TSC3.make_self_consistent(5,0.);

    // lmn_orb LMN2(&Pot,0.01,500.);
    // df_double_power DF4({1.,1.,1.3},100.,3.01,0.,&Pot,&LMN2,{},1.5e7/conv::G);
    // TriaxialSelfConsistent TSC4(&DF4,&Pot,"nfw_start/200215_1_1_13_finerres",1.,0.01,100.,20,6,6);
    // TSC4.make_self_consistent(5,0.);

    // Isochrone Pot(12.3e5,5.,0.995,0.99);


    // lmn_orb LMN3(&Pot,0.01,500.);
    // df_double_power DF5({1.,0.7,1.},100.,3.01,0.,&Pot,&LMN3,{},1.5e7/conv::G);
    // TriaxialSelfConsistent TSC5(&DF5,&Pot,"nfw_start/200215_1_07_1_expgrid",1.,0.01,100.,20,6,6);
    // TSC5.make_self_consistent(5,0.);

    // lmn_orb LMN4(&Pot,0.01,500.);
    // df_double_power DF6({1.3,1.,1.},100.,3.01,0.,&Pot,&LMN4,{},1.5e7/conv::G);
    // TriaxialSelfConsistent TSC6(&DF6,&Pot,"nfw_start/200215_13_1_1",1.,0.01,100.,20,6,6);
    // TSC6.make_self_consistent(5,0.);

    // lmn_orb LMN5(&Pot,0.01,500.);
    // df_double_power DF7({1.,0.9,1.2},100.,3.01,0.,&Pot,&LMN5,{},1.5e7/conv::G);
    // TriaxialSelfConsistent TSC7(&DF7,&Pot,"nfw_start/200215_1_09_12",1.,0.01,100.,20,6,6);
    // TSC7.make_self_consistent(5,0.);

    // lmn_orb LMN6(&Pot,0.01,500.);
    // df_double_power DF8({.5,1.,1.},100.,3.01,0.,&Pot,&LMN6,{},1.5e7/conv::G);
    // TriaxialSelfConsistent TSC8(&DF8,&Pot,"nfw_start/200215_05_1_1",1.,0.01,100.,20,6,6);
    // TSC8.make_self_consistent(5,0.);





    // Isochrone Pot(2e6,6.,1.,1.);
    // df_isochrone DFI({1.,1.,1.},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC(&DFI,&Pot,"iso/iso_1_1_1_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // df_isochrone DFI2({1.,1.,1.2},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC2(&DFI2,&Pot,"iso/iso_1_1_12_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // df_isochrone DFI3({1.,1.,1.3},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC3(&DFI3,&Pot,"iso/iso_1_1_13_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // df_isochrone DFI4({1.,1.,1.4},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC4(&DFI4,&Pot,"iso/iso_1_1_14_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // df_isochrone DFI5({1.,1.,1.5},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC5(&DFI5,&Pot,"iso/iso_1_1_15_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // df_isochrone DFI6({1.,1.,1.6},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC6(&DFI6,&Pot,"iso/iso_1_1_16_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // df_isochrone DFI7({1.,1.1,1.2},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC7(&DFI7,&Pot,"iso/iso_1_11_12_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // df_isochrone DFI8({1.,.9,1.2},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC8(&DFI8,&Pot,"iso/iso_1_09_12_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // df_isochrone DFI9({1.,0.7,1.2},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC9(&DFI9,&Pot,"iso/iso_1_07_12_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    //     df_isochrone DFI10({1.,0.7,1.2},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC10(&DFI10,&Pot,"iso/iso_1_07_12_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    //     df_isochrone DFI11({1.,0.7,1.4},2e6,6.,&Pot,1);
    // TriaxialSelfConsistent TSC11(&DFI11,&Pot,"iso/iso_1_07_14_ujc",1.);
    // TSC.make_self_consistent(10,0.);

    // IsochronePotential Pot_Sph(2e6,6.);
    // df_isochrone_spherical DFI_Sph(2e6,6.,&Pot_Sph);
    // TriaxialSelfConsistent TSC_Sph(&DFI_Sph,&Pot_Sph,"iso/iso_sph",6.);
