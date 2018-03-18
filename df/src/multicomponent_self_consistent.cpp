// ============================================================================
// Main for building self-consistent models
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
#include "genfunc_aa.h"
#include "uv_orb.h"
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
#include <stdlib.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

const std::string data_folder = "data/";

struct config_params{

    std::vector<std::string> models;
    VecDoub a0,inr,our;
    VecInt nr,na,lmax;
    MatDoub params;
    VecInt npars;
    double irmin,ormax;
    unsigned ncompt;

    config_params(std::string config_file,std::string type){
        std::ifstream config; config.open(config_file);
        int shift=-4;
        if(type=="axisymmetric" or type=="triaxial")
            shift-=2;
        std::string line;double b;std::string n;
        while(std::getline(config, line)){
            VecDoub pars;
            std::stringstream s(line);
            s>>n; models.push_back(n);
            while(s>>b) pars.push_back(b);
            params.push_back(pars);
            npars.push_back(pars.size()+shift);
        }
        ncompt = params.size();
        for(auto i:params)a0.push_back(i[i.size()+shift]);shift++;
        for(auto i:params)inr.push_back(i[i.size()+shift]);shift++;
        for(auto i:params)our.push_back(i[i.size()+shift]);shift++;
        for(auto i:params)nr.push_back(i[i.size()+shift]);shift++;
        if(type=="axisymmetric" or type=="triaxial"){
            for(auto i:params)na.push_back(i[i.size()+shift]);shift++;
            for(auto i:params)lmax.push_back(i[i.size()+shift]);
        }
        for(unsigned i=0;i<params.size();++i) params[i].resize(npars[i]);
        irmin=Min(inr);ormax=Max(our);
        printMatrix(params);
    }
};

// ============================================================================

int main(int argc, char*argv[]){

    // Need at least a model type, symmetry and filename
    if(argc<4) exit(-1);

    setenv("CUBACORES","0",1);
    gsl_set_error_handler_off();

    // different options -- default is build, but if environment variables
    // found it will do different stuff
    auto veldisp = getenv("VELDISP");
    int find_veldisp =0;
    if(veldisp!=nullptr)
        find_veldisp = atoi(veldisp);

    auto split = getenv("SPLIT");
    int find_splitdensity =0;
    if(split!=nullptr)
        find_splitdensity = atoi(split);

    auto proj = getenv("PROJ");
    int find_projdensity =0;
    if(proj!=nullptr)
        find_projdensity = atoi(proj);

    auto highres = getenv("HIGHRES");
    int find_highresdensity =0;
    if(highres!=nullptr)
        find_highresdensity = atoi(highres);

    auto genfunc = getenv("GENFUNC");
    int use_genfunc=0;
    if(genfunc!=nullptr)
        use_genfunc = atoi(genfunc);

    auto densityl = getenv("DENSITY");
    int oned_density=0;
    if(densityl!=nullptr)
        oned_density = atoi(densityl);

    auto blkhll = getenv("BLACKHOLE");
    int blackhole=0;
    if(blkhll!=nullptr)
        blackhole = atoi(blkhll);

    Potential_JS *Pot;
    SphericalPotential *SphrPot;
    Action_Finder *ActFinder;

    Potential_JS *ExtPot = nullptr;
    SphericalPotential *SphrExtPot = nullptr;

    bool is_spherical=false, is_axisymmetric=false, is_triaxial=false;

    std::string symmetry = std::string(argv[1]);
    if(symmetry=="spherical")
        is_spherical=true;
    else if(symmetry=="axisymmetric")
        is_axisymmetric=true;
    else if(symmetry=="triaxial")
        is_triaxial=true;
    else{
        std::cerr<<"Symmetry not recognised: try spherical, axisymmetric or triaxial"<<std::endl;
        return 0;
    }

    if(blackhole){
        if(symmetry=="spherical")
            SphrExtPot = new PowerLawSpherical(0.001,1.);
        else
            ExtPot = new PowerLaw(0.001,1.);
    }
    // Read model type
    std::string delta_file = data_folder+std::string(argv[2]);
    std::string config_file = std::string(argv[3]);

    config_params CP(config_file,symmetry);

    if(CP.models[0]=="WE"){
        if(is_spherical){
            SphrPot = new HernquistSpherical(1.,1.);
            ActFinder = new Actions_Spherical(SphrPot);
        }else if(is_axisymmetric){
            Pot = new Hernquist(1.,1.,1.,0.95);
            ActFinder = new uv_orb(Pot,CP.irmin*1.05,CP.ormax*0.95,16,5,delta_file);
        }else{
            Pot = new Hernquist(1.,1.,0.98,0.95);
            if(use_genfunc)
                ActFinder = new Actions_Genfunc(Pot);
            else
                ActFinder = new lmn_orb(Pot,CP.irmin*1.05,CP.ormax*0.95,16,true,false,delta_file);
        }
    }
    else if(CP.models[0]=="PB"){
        if(is_spherical){
                SphrPot = new HernquistSpherical(1.,1.);
                ActFinder = new Actions_Spherical(SphrPot);
            }else if(is_axisymmetric){
                Pot = new Hernquist(1.,1.,1.,0.95);
                ActFinder = new uv_orb(Pot,CP.irmin*1.05,CP.ormax*0.95,16,5,delta_file);
            }else{
                Pot = new Hernquist(1.,1.,0.999,0.995);
                ActFinder = new lmn_orb(Pot,CP.irmin*1.05,CP.ormax*0.95,16,true,false,delta_file);
            }
    }
    else if(CP.models[0]=="Double"){
        if(is_spherical){
            if(std::string(argv[11])=="Hernquist")
                SphrPot = new HernquistSpherical(1.,1.);
            else
                SphrPot = new NFWSpherical(1.,1.);
            ActFinder = new Actions_Spherical(SphrPot);
        }else if(is_axisymmetric){
            Pot = new NFW(1.,1.,1.,0.999);
            ActFinder = new uv_orb(Pot,CP.irmin*1.05,CP.ormax*0.95,16,5,delta_file);
        }else{
            if(std::string(argv[11])=="Hernquist")
                Pot = new Hernquist(.3,1.,0.999,0.995);
            else
                Pot = new NFW(1.,1.,1.,1.);
            ActFinder = new lmn_orb(Pot,CP.irmin*1.05,CP.ormax*0.95,16,true,false,delta_file);
        }
    }
    else if(CP.models[0]=="Isochrone"){
        if(is_spherical){
                SphrPot = new IsochronePotential(CP.params[0][0],CP.params[0][1]);
                ActFinder = new Actions_Spherical(SphrPot);
            }else if(is_axisymmetric){
                Pot = new Isochrone(CP.params[0][0],CP.params[0][1],1.,0.9995);
                ActFinder = new uv_orb(Pot,CP.irmin*1.05,CP.ormax*0.95,16,5,delta_file);
            }else{
                Pot = new Isochrone(CP.params[0][0],CP.params[0][1],0.9999,0.9995);
                ActFinder = new lmn_orb(Pot,CP.irmin*1.05,CP.ormax*0.95,16,true,false,delta_file);
            }
    }
    else{
        std::cerr<<"Model 0 not recognised: try WE, PB, Double or Isochrone"<<std::endl;
        return 0;
    }

    df_multicomponent* DFm = new df_multicomponent(Pot,ActFinder);

    for(unsigned kk=0;kk<CP.ncompt;++kk){
        if(CP.models[kk]=="WE"){
            assert(CP.npars[kk]==11);
            DFm->add_new_df(new df_we(CP.params[kk][0],CP.params[kk][1],
                                   CP.params[kk][2],CP.params[kk][3],
                                   CP.params[kk][4],CP.params[kk][5],
                                   CP.params[kk][6],CP.params[kk][7],
                                   is_spherical?SphrPot:Pot,ActFinder,
                                   CP.params[kk][10],0.0,
                                   CP.params[kk][8],CP.params[kk][9],
                                   symmetry
                                   ));
        }
        else if(CP.models[kk]=="PB"){
            assert(CP.npars[kk]==4);
            DFm->add_new_df(new df_pb(CP.params[kk][0],CP.params[kk][1],
                                   CP.params[kk][2],symmetry,
                  is_spherical?SphrPot:Pot,ActFinder,CP.params[kk][3],0.0));
        }
        else if(CP.models[kk]=="Double"){
            assert(CP.npars[kk]==4);
            DFm->add_new_df(
              new df_double_power({CP.params[kk][0],CP.params[kk][1],
                                   CP.params[kk][2]},CP.params[kk][3],
                                   CP.params[kk][4],CP.params[kk][5],
                                   CP.params[kk][6],symmetry,
                                   is_spherical?SphrPot:Pot,ActFinder,{},
                                   CP.params[kk][7],0.0));
        }
        else if(CP.models[kk]=="Isochrone"){
            assert(CP.npars[kk]==4);
            DFm->add_new_df(
                new df_isochrone({ CP.params[kk][0],CP.params[kk][1]},
                                   CP.params[kk][2],CP.params[kk][3],
                                   symmetry,
                                   is_spherical?SphrPot:Pot,ActFinder));
        }
        else{
            std::cerr<<"Model "<<kk<<" not recognised: try WE, PB, Double or Isochrone"<<std::endl;
            return 0;
        }
    }
    // Now find model or calculate the velocity dispersions
    if(is_spherical){
        if(find_veldisp){
            SphericalMultiComSelfConsistent SCM(DFm,argv[2],SphrExtPot);
            SCM.xidisp(data_folder+std::string(argv[2])+".xdisp",0.01,100.,40,"log",5e-4,1);
        }
        else{
            SphericalMultiComSelfConsistent SCM(DFm,SphrPot,argv[2],CP.a0,CP.inr,CP.our,CP.nr,SphrExtPot);
            SCM.make_self_consistent(100,0.);
        }
    }
    else if(is_axisymmetric){
        if(find_veldisp){
            AxisymmetricMultiComSelfConsistent SCM(DFm,argv[2],ExtPot);
            SCM.xidisp(data_folder+std::string(argv[2])+".xdisp",0.1,100.,8.);
        }
        else{
            AxisymmetricMultiComSelfConsistent SCM(DFm,Pot,argv[2],CP.a0,CP.inr,CP.our,CP.nr,CP.na,CP.lmax,ExtPot);
            SCM.make_self_consistent(100,0.);
        }
    }
    else if(is_triaxial){
        if(find_veldisp){
            TriaxialMultiComSelfConsistent SCM(DFm,argv[2],ExtPot);
            SCM.xidisp(data_folder+std::string(argv[2])+".xdisp",0.05,10.,12,"log",2e-3);
        }
        else if(find_splitdensity){
            TriaxialMultiComSelfConsistent SCM(DFm,argv[2],ExtPot);
            SCM.split_density(data_folder+std::string(argv[2])+".split",0.015,200.,30,"log",2e-3);
        }
        else if(find_projdensity){
            TriaxialMultiComSelfConsistent SCM(DFm,argv[2],ExtPot);
            SCM.projected_density(data_folder+std::string(argv[2])+".proj",2.,30,8e-4);
        }
        else if(find_highresdensity){
            TriaxialMultiComSelfConsistent SCM(DFm,argv[2],ExtPot);
            SCM.density(data_folder+std::string(argv[2])+".highres",0.01,22.,30,"log",2e-3);
        }
        else if(use_genfunc){
            TriaxialMultiComSelfConsistent SCM(DFm,argv[2],ExtPot);
            SCM.oneD_density(data_folder+std::string(argv[2])+".genfunc_test2",0.01,22.,30,"log",1e-2);
        }
        else if(oned_density){
            TriaxialMultiComSelfConsistent SCM(DFm,argv[2],ExtPot);
            SCM.oneD_density(data_folder+std::string(argv[2])+".fudge",0.01,22.,30,"log",5e-3);
        }
        else{
            TriaxialMultiComSelfConsistent SCM(DFm,Pot,argv[2],CP.a0,CP.inr,CP.our,CP.nr,CP.na,CP.lmax,ExtPot);
            SCM.make_self_consistent(10,0.);
        }
    }
}

// ============================================================================
