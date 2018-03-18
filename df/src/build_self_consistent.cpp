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
// #include "lmn_orb_ELLz.h"
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
// Some of the models look scruffy in the outskirts
// Could alter lmn_orb to make alpha and beta defaults better
// ============================================================================

int main(int argc, char*argv[]){

    std::cout<<"Using units where G="<<conv::G<<std::endl;

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

    df *DF;
    Potential_JS *Pot;
    SphericalPotential *SphrPot;
    Action_Finder *ActFinder;
    Potential_JS *ExtPot = nullptr;
    SphericalPotential *SphrExtPot = nullptr;
    lmn_orb *Acttmp;

    bool is_spherical=false, is_axisymmetric=false, is_triaxial=false;

    std::string symmetry = std::string(argv[2]);
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
    std::string model = std::string(argv[1]);
    std::string delta_file = data_folder+std::string(argv[3]);

    if(model=="WE"){
        if(is_spherical){
            SphrPot = new HernquistSpherical(1.,1.);
            ActFinder = new Actions_Spherical(SphrPot);
        }else if(is_axisymmetric){
            Pot = new Hernquist(1.,1.,1.,0.95);
            ActFinder = new uv_orb(Pot,0.1,200.,16,5,delta_file);
        }else{
            Pot = new Hernquist(1.,1.,0.98,0.95);
            if(use_genfunc){
                Acttmp = new lmn_orb(Pot,.015,180.,24,true,false,delta_file);
                ActFinder = new Actions_Genfunc(Pot,symmetry,Acttmp,false);
            }
            else
                ActFinder = new lmn_orb(Pot,.015,180.,24,true,false,delta_file);
        }
        DF = new df_we( atof(argv[4]),atof(argv[5]),atof(argv[6]),
                        atof(argv[7]),atof(argv[8]),atof(argv[9]),
                        atof(argv[10]),atof(argv[11]),
                        is_spherical?SphrPot:Pot,ActFinder,1.,0.0,
                        atof(argv[12]),atof(argv[13]),
                        symmetry
                        );
    }
    else if(model=="PB"){
        if(is_spherical){
                SphrPot = new HernquistSpherical(1.,1.);
                ActFinder = new Actions_Spherical(SphrPot);
            }else if(is_axisymmetric){
                Pot = new Hernquist(1.,1.,1.,0.95);
                ActFinder = new uv_orb(Pot,0.1,200.,16,5,delta_file);
            }else{
                Pot = new Hernquist(1.,1.,0.999,0.995);
                ActFinder = new lmn_orb(Pot,.015,180.,16,false,false,delta_file);
            }
        DF = new df_pb(atof(argv[4]),atof(argv[5]),atof(argv[6]),symmetry,
                  is_spherical?SphrPot:Pot,ActFinder,1.,0.0);
    }
    else if(model=="Double"){
        if(is_spherical){
            if(std::string(argv[11])=="Hernquist")
                SphrPot = new HernquistSpherical(1.,1.);
            else
                SphrPot = new NFWSpherical(1.,1.);
            ActFinder = new Actions_Spherical(SphrPot);
        }else if(is_axisymmetric){
            Pot = new NFW(1.,1.,1.,0.999);
            ActFinder = new uv_orb(Pot,0.1,200.,16,5,delta_file);
        }else{
            if(std::string(argv[11])=="Hernquist")
                Pot = new Hernquist(.3,1.,0.999,0.995);
            else
                Pot = new NFW(1.,1.,1.,1.);
            ActFinder = new lmn_orb(Pot,.015,180.,16,false,false,delta_file);
        }
        DF = new df_double_power({atof(argv[4]),atof(argv[5]),atof(argv[6])},
                                  atof(argv[7]),atof(argv[8]),atof(argv[9]),
                                  atof(argv[10]),symmetry,
                                  is_spherical?SphrPot:Pot,ActFinder,{},
                                  1.,0.0);
    }
    else if(model=="Isochrone"){
        if(is_spherical){
                SphrPot = new IsochronePotential(atof(argv[6]),atof(argv[7]));
                ActFinder = new Actions_Spherical(SphrPot);
            }else if(is_axisymmetric){
                Pot = new Isochrone(atof(argv[6]),atof(argv[7]),1.,0.9995);
                ActFinder = new uv_orb(Pot,0.02,180.,16,5,delta_file);
            }else{
                Pot = new Isochrone(atof(argv[6]),atof(argv[7]),0.995,0.99);
                if(use_genfunc){
                    Acttmp = new lmn_orb(Pot,.015,180.,24,true,false,delta_file);
                    ActFinder = new Actions_Genfunc(Pot,symmetry,Acttmp,false);
                    // ActFinder = new Actions_Genfunc_Average(Pot,symmetry,Acttmp,false);
                }
                else
                    ActFinder = new lmn_orb(Pot,.015,180.,24,true,false,delta_file);
                }
        DF = new df_isochrone({ atof(argv[4]),atof(argv[5])},
                                atof(argv[6]),atof(argv[7]),
                                symmetry,
                                is_spherical?SphrPot:Pot,ActFinder);
    }
    else{
        std::cerr<<"Model not recognised: try WE, PB, Double or Isochrone"<<std::endl;
        return 0;
    }
    // Now find model or calculate the velocity dispersions
    if(is_spherical){
        if(find_veldisp){
            SphericalSelfConsistent SCM(DF,argv[3],SphrExtPot);
            SCM.xidisp(data_folder+std::string(argv[3])+".xdisp",0.01,100.,40,"log",5e-4,1);
        }
        else{
            SphericalSelfConsistent SCM(DF,SphrPot,argv[3],1.,0.005,100.,80,SphrExtPot);
            SCM.make_self_consistent(100,0.);
        }
    }
    else if(is_axisymmetric){
        if(find_veldisp){
            AxisymmetricSelfConsistent SCM(DF,argv[3],ExtPot);
            SCM.xidisp(data_folder+std::string(argv[3])+".xdisp",0.1,100.,8.);
        }
        else{
            AxisymmetricSelfConsistent SCM(DF,Pot,argv[3],1.,0.01,200.,50,8,8,ExtPot);
            SCM.make_self_consistent(100,0.);
        }
    }
    else if(is_triaxial){
        if(find_veldisp){
            TriaxialSelfConsistent SCM(DF,argv[3],ExtPot);
            SCM.xidisp(data_folder+std::string(argv[3])+".xdisp2",0.05,10.,12,"log",5e-3);
        }
        else if(find_splitdensity){
            TriaxialSelfConsistent SCM(DF,argv[3],ExtPot);
            SCM.split_density(data_folder+std::string(argv[3])+".split_new",0.015,200.,30,"log",5e-4);
        }
        else if(find_projdensity){
            TriaxialSelfConsistent SCM(DF,argv[3],ExtPot);
            SCM.projected_density(data_folder+std::string(argv[3])+".proj",2.,30,1e-4);
        }
        else if(find_highresdensity){
            TriaxialSelfConsistent SCM(DF,argv[3],ExtPot);
            SCM.density(data_folder+std::string(argv[3])+".highres",0.01,22.,30,"log",2e-3);
        }
        else if(use_genfunc){
            TriaxialSelfConsistent SCM(DF,argv[3],ExtPot);
            SCM.oneD_density(data_folder+std::string(argv[3])+".genfunc_take2",0.01,22.,30,"log",5e-3);
        }
        else if(oned_density){
            TriaxialSelfConsistent SCM(DF,argv[3],ExtPot);
            SCM.oneD_density(data_folder+std::string(argv[3])+".fudge",0.01,22.,30,"log",5e-3);
        }
        else{
            TriaxialSelfConsistent SCM(DF,Pot,argv[3],1.,0.01,200.,40,8,8,ExtPot);
            SCM.make_self_consistent(15,0.);
        }
    }
}

// ============================================================================
