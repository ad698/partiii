// ============================================================================
// Main for calculating velocity dispersions of self-consistent models
// ============================================================================

#include <iostream>
#include <vector>
#include <memory>
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
#include "spherical_aa.h"
#include "stackel_aa.h"
#include "uv_orb.h"
#include "lmn_orb.h"
#include "cuba/cuba.h"
#include "cubature/cubature.h"
#include <ctime>
#include <string>
#include <thread>
#include "df.h"
#include "moments.h"
#include "Multipole.h"
#include "self_consistent.h"
#include <stdlib.h>

const std::string data_folder = "data/";

// ============================================================================

int main(int argc, char*argv[]){

    setenv("CUBACORES","0",1);

    if(std::string(argv[8])=="triaxial"){
        // We don't use this potential but just pass as a placeholder
    	NFW Pot(.3,1.,0.999,0.995);
        lmn_orb ActFinder(&Pot,.2,100.,16);
        df_double_power DF({atof(argv[1]),atof(argv[2]),atof(argv[3])},
                             atof(argv[4]),atof(argv[5]),atof(argv[6]),1.,
                             &Pot,&ActFinder,{},1.,0.0);
        TriaxialSelfConsistent SCM(&DF,std::string(argv[7])+".me");
        SCM.xidisp(data_folder+std::string(argv[7])+".xdisp",0.1,100.,8.);
    }
    else if(std::string(argv[8])=="axisymmetric"){
        // We don't use this potential but just pass as a placeholder
        NFW Pot(.3,1.,1.,0.999);
        uv_orb ActFinder(&Pot,0.01,200.,16);
        df_double_power DF({atof(argv[1]),atof(argv[2]),atof(argv[3])},
                             atof(argv[4]),atof(argv[5]),atof(argv[6]),1.,
                             &Pot,&ActFinder,{},1.,0.0);
        SphericalSelfConsistent SCM(&DF,std::string(argv[7])+".me");
        SCM.xidisp(data_folder+std::string(argv[7])+".xdisp",0.1,100.,8.);
    }
    else if(std::string(argv[8])=="spherical"){
        // We don't use this potential but just pass as a placeholder
        NFWSpherical Pot(.3,1.);
        Actions_Spherical ActFinder(&Pot);
        df_double_power DF({atof(argv[1]),atof(argv[2]),atof(argv[3])},
                             atof(argv[4]),atof(argv[5]),atof(argv[6]),1.,
                             &Pot,&ActFinder,{},1.,0.0);
        SphericalSelfConsistent SCM(&DF,std::string(argv[7])+".me");
        SCM.xidisp(data_folder+std::string(argv[7])+".xdisp",0.1,100.,8.);
    }
}

// ============================================================================
