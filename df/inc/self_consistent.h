#ifndef SELFCON_H
#define SELFCON_H

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

#include <vector>
#include "utils.h"
#include "potential.h"
#include "df.h"
#include "moments.h"
#include "Multipole.h"

// ============================================================================

class Density_from_DF: public Density{
    /* Wrapper for DF to calculate density that we can pass to Multipole */
private:
    df *DF;
public:
    Density_from_DF(df *DF):Density(0.),DF(DF){};
    inline double density(const VecDoub& x){
        double IE = 5e-3;
        if(norm(x)<1.) IE = 2e-3;
        return DF->density(x,IE);}
};

// ============================================================================

class IncrementalPotential: public Potential_JS{
    /* Takes two potentials and adds with weight */
private:
    Potential_JS *oldpot, *newpot;
    double gamma;
    Potential_JS *extpot;
public:
    IncrementalPotential(void){};
    virtual ~IncrementalPotential(void){};
    IncrementalPotential(Potential_JS *oldpot, Potential_JS *newpot,double gamma, Potential_JS *extpot=nullptr)
        :oldpot(oldpot), newpot(newpot), gamma(gamma), extpot(extpot){};

    double Phi(const VecDoub& x){
        if(gamma==0.) return newpot->Phi(x)+(extpot?extpot->Phi(x):0.);
        else
            return (1+gamma)*(newpot->Phi(x)+(extpot?extpot->Phi(x):0.))
                   -gamma*oldpot->Phi(x);
    }
    VecDoub Forces(const VecDoub& x){
        if(gamma>0.)
            return newpot->Forces(x)+(extpot?extpot->Forces(x):VecDoub(3,0.));
        else
            return (newpot->Forces(x)+(extpot?extpot->Forces(x):VecDoub(3,0.)))*(1+gamma)-oldpot->Forces(x)*gamma;
    }
};

class IncrementalSphericalPotential: public SphericalPotential{
    /* Takes two spherical potentials and adds with weight */
private:
    SphericalPotential *oldpot;
    Potential_JS *newpot;
    double gamma;
    SphericalPotential *extpot;
public:
    IncrementalSphericalPotential(void){};
    virtual ~IncrementalSphericalPotential(void){};
    IncrementalSphericalPotential(SphericalPotential *oldpot, Potential_JS *newpot,double gamma, SphericalPotential *extpot=nullptr)
        :oldpot(oldpot), newpot(newpot), gamma(gamma),extpot(extpot){};

    double Phi_r(double r){
        if(gamma==0.)
            return newpot->Phi({r,0.,0.})+(extpot?extpot->Phi_r(r):0.);
        else
            return (1+gamma)*(newpot->Phi({r,0.,0.})
                              +(extpot?extpot->Phi_r(r):0.))
                    -gamma*oldpot->Phi_r(r);
    }
    double dPhi_r(double r){
        if(gamma==0.) return -newpot->Forces({r,0.,0.})[0]+(extpot?extpot->dPhi_r(r):0.);
        else return -(1+gamma)*newpot->Forces({r,0.,0.})[0]-gamma*oldpot->dPhi_r(r)+(1+gamma)*(extpot?extpot->dPhi_r(r):0.);
    }
};

// class ResidualDiscPotential: public MultipoleExpansion_Axisymmetric{

// };

// class ResidualDiscDensity_from_DF: public Density{
//     /* Wrapper for DF to calculate density that we can pass to Multipole */
// private:
//     df *DF;
// public:
//     Density_from_DF(df *DF):Density(0.),DF(DF){};
//     inline double density(const VecDoub& x){
//         double IE = 5e-3;
//         if(norm(x)<1.) IE = 2e-3;
//         return DF->density(x,IE)-;}
// };

// ============================================================================

namespace selfcon{

    inline void stamplogfile(std::string filename,int n){
        // Write converged to end of log file
        std::ofstream outfile;
        outfile.open(filename+".log", std::ios_base::app | std::ios_base::out);
        outfile<<"Converged at iteration "<<n<<std::endl;
        outfile.close();
    }

    inline void printMmaxtolog(std::string filename,int n,double M){
        // Write converged to end of log file
        std::ofstream outfile;
        outfile.open(filename+".log", std::ios_base::app | std::ios_base::out);
        outfile<<"Mass enclosed after "<<n<<" iteration = "<<M<<std::endl;
        outfile.close();
    }

    template<class c>
    void writelogfile(c *C, std::string filename){
        std::ofstream outfile; outfile.open(filename+".log");
        outfile<<C->name()<<"\n\n"<<C->params()<<std::endl;
        outfile.close();
    }

    void output_pot(VecDoub radial_grid,Potential_JS *InitPot,std::string name,std::string type);

    inline void output_initial_pot(VecDoub radial_grid,Potential_JS *InitPot,std::string name,std::string type){
        return output_pot(radial_grid,InitPot,name+"_initpot.vis",type);
    }

    inline void output_ext_pot(VecDoub radial_grid,Potential_JS *InitPot,std::string name,std::string type){
        return output_pot(radial_grid,InitPot,name+"_extpot.vis",type);
    }

    inline void output_total_pot(VecDoub radial_grid,Potential_JS *InitPot,int n,std::string name,std::string type){
        return output_pot(radial_grid,InitPot,name+"_totalpot"+std::to_string(n)+".vis",type);
    }
}

// ============================================================================

class SphericalSelfConsistent: public df{
    /* Calculates a self-consistent model for spherical symmetry */
    /* Note that the action-finding object to be passed to the df*/
    /* must be spherical -- this could be made automatic...      */
private:

    const std::string data_folder = "data/";

    df *DF;
    MultipoleExpansion_Spherical *ME;
    Density_from_DF DDD;
    SphericalPotential *InitPot;

    std::string name;
    double a0;
    double innerradius, outerradius;
    const double threshold = 0.01; // quits when all potential devs less than

    SphericalPotential *ExternalPot;

public:

    // This code is used for finding the properties of the model after
    // construction
    SphericalSelfConsistent(df *DF, std::string MEfile, SphericalPotential *exPot=nullptr)
        :df(ME,DF->ActionCalculator),DF(DF),
         ME(new MultipoleExpansion_Spherical(data_folder+MEfile+".me")),
         DDD(Density_from_DF(DF)){
            IncrementalSphericalPotential *Sphrpot = new IncrementalSphericalPotential(InitPot,ME,0.);
            pot=Sphrpot;
            ActionCalculator->reset_sph(Sphrpot);
            DF->reset_pot(Sphrpot);
            a0=ME->scaleradius();
            innerradius=ME->innerradius();
            outerradius=ME->outerradius();
    }

    inline double action_dist(const VecDoub& x){ return DF->action_dist(x);}
    inline double realspace_dist(const VecDoub& x){
        return DF->realspace_dist(x);}
    inline VecDoub realspace_dist_vec(const VecDoub& x){
        return DF->realspace_dist_vec(x);}

    // This code is used to build a model

    SphericalSelfConsistent(df *DF, SphericalPotential *InitPot, std::string Name, double a0 = 1., double irr=0.1, double orr=500., int nr=15, SphericalPotential *ExternalPot=nullptr)
        :df(ME),DF(DF),
        DDD(Density_from_DF(DF)),InitPot(InitPot),
        name(data_folder+Name),a0(a0),innerradius(irr),outerradius(orr),
        ExternalPot(ExternalPot)
        {
        std::cout <<"---------------------------------------------\n";
        std::cout <<"Constructing the spherical self-consistent DF given by:\n"
                  <<DF->name()<<"\n\twith params "<<DF->params()
                  <<"\nwith an initial potential given by:\n"
                  <<InitPot->name()<<"\nwith params "<<InitPot->params()
                  <<"\nusing a multipole expansion with NR = "<<nr<<
                  ", a0 = "<<a0<<", ir = "<<irr
                  <<", or = "<<orr<<std::endl;
        if(ExternalPot)
            std::cout<<"and external potential: "<<ExternalPot->params()<<"\n";
        std::cout <<"---------------------------------------------\n";

        selfcon::writelogfile(DF,name);


        ME = new MultipoleExpansion_Spherical(&DDD,
                                        nr,/* number of radial grid points */
                                        a0,/* scale radius */
                                        irr,orr);
        outerradius=ME->outerradius();
        // Output initial properties
        ME->visualize(name+".vis");
        ME->output_to_file(name+".me");
	selfcon::printMmaxtolog(name,0,ME->Mass(orr));
        selfcon::output_initial_pot(ME->get_radial_grid(),InitPot,name,"spherical");
        if(ExternalPot)
            selfcon::output_ext_pot(ME->get_radial_grid(),ExternalPot,name,"spherical");
    }
    void make_self_consistent(unsigned NITER, double gamma = 0.5);
};

class SphericalMultiComSelfConsistent: public df{
    /* Calculates a self-consistent model for spherical symmetry */
    /* Note that the action-finding object to be passed to the df*/
    /* must be spherical -- this could be made automatic...      */
private:

    const std::string data_folder = "data/";

    df_multicomponent *DFs;
    std::vector<MultipoleExpansion_Spherical*> ME;
    std::vector<Density_from_DF> DDD;

    SphericalPotential *InitPot;
    MultiComponentPotential<MultipoleExpansion_Spherical> *TotalPot;

    std::string name;
    VecInt NR;
    VecDoub a0,innerradius, outerradius;
    const double threshold = 0.01; // quits when all potential devs less than

    SphericalPotential *ExternalPot;

    VecDoub radial_grid;

public:

    void fill_radial_grid(void){
        double mini = Min(innerradius), maxo = Max(outerradius);
        double minstep = 1e9;
        for(unsigned i=0;i<DFs->ncompts();++i){
            double s = log(outerradius[i]/innerradius[i])/(double)NR[i];
            if(s<minstep)
                minstep=s;
        }
        for(double ss=log(mini);ss<log(maxo);ss+=minstep)
            radial_grid.push_back(exp(ss));
    }

    // This code is used for finding the properties of the model after
    // construction
    SphericalMultiComSelfConsistent(df_multicomponent *DFs, std::string MEfile, SphericalPotential *exPot=nullptr)
        :df(TotalPot,DFs->ActionCalculator),DFs(DFs)
         {
            for(unsigned j=0;j<DFs->ncompts();++j){
                ME.push_back(new MultipoleExpansion_Spherical(data_folder+MEfile+".me"+std::to_string(j)));
                DDD.push_back(Density_from_DF(DFs->compt(j)));
                a0.push_back((ME[j])->scaleradius());
                innerradius.push_back((ME[j])->innerradius());
                outerradius.push_back((ME[j])->outerradius());
            }
            TotalPot = new MultiComponentPotential<MultipoleExpansion_Spherical>(ME);
            IncrementalSphericalPotential *Sphrpot = new IncrementalSphericalPotential(InitPot,TotalPot,0.,ExternalPot);
            pot=Sphrpot;
            ActionCalculator->reset_sph(Sphrpot);
            DFs->reset_pot(Sphrpot);

    }

    inline double action_dist(const VecDoub& x){ return DFs->action_dist(x);}
    inline double realspace_dist(const VecDoub& x){
        return DFs->realspace_dist(x);}
    inline VecDoub realspace_dist_vec(const VecDoub& x){
        return DFs->realspace_dist_vec(x);}

    // This code is used to build a model

    SphericalMultiComSelfConsistent(df_multicomponent *DFs, SphericalPotential *InitPot, std::string Name, VecDoub a0, VecDoub irr, VecDoub orr, VecInt nr, SphericalPotential *ExternalPot=nullptr)
        :df(TotalPot),DFs(DFs),InitPot(InitPot)
        ,name(data_folder+Name),NR(nr),a0(a0),innerradius(irr),outerradius(orr),ExternalPot(ExternalPot){

        std::cout<<"---------------------------------------------\n";
        std::cout<<"Constructing a multicomponent spherical self-consistent DF given by:\n"
                 <<DFs->name()<<"\nwith params "<<DFs->params()
                 <<"\nwith an initial potential given by:\n"
                 <<InitPot->name()<<"\nwith params "<<InitPot->params()
                 <<"\nusing multipole expansion with ";
        for(unsigned i=0;i<DFs->ncompts();++i)
            std::cout<<std::to_string(i+1)+". NR = "<<nr[i]
                     <<", a0 = "<<a0[i]<<", ir = "<<irr[i]
                     <<", or = "<<orr[i]<<std::endl;
        if(ExternalPot)
            std::cout<<"and external potential: "<<ExternalPot->params()<<"\n";
        std::cout<<"---------------------------------------------\n";

        selfcon::writelogfile(DFs,name);

        fill_radial_grid();

        for(unsigned i=0;i<DFs->ncompts();++i){
            DDD.push_back(Density_from_DF(DFs->compt(i)));
            ME.push_back(new MultipoleExpansion_Spherical(&(DDD[i]),
                            nr[i],/* number of radial grid points */
                            a0[i],/* scale radius */
                            irr[i],orr[i]));
            ME[i]->output_to_file(name+".me"+std::to_string(i));
            ME[i]->visualize(name+".vis"+std::to_string(i));
        }

        fill_radial_grid();

        TotalPot = new MultiComponentPotential<MultipoleExpansion_Spherical>(ME);

        selfcon::output_initial_pot(radial_grid,InitPot,name,"spherical");
        if(ExternalPot)
            selfcon::output_ext_pot(radial_grid,ExternalPot,name,"spherical");
    }
    void make_self_consistent(unsigned NITER, double gamma = 0.5);
};

// ============================================================================

class AxisymmetricSelfConsistent: public df{
    /* Calculates a self-consistent model */
private:

    const std::string data_folder = "data/";

    df *DF;
    MultipoleExpansion_Axisymmetric *ME;
    Density_from_DF DDD;
    Potential_JS *InitPot;
    std::string name;
    double a0;
    double innerradius, outerradius;
    const double threshold = 0.01; // quits when all potential devs less than

    Potential_JS *ExternalPot;
public:

    // This code is used for finding the properties of the model after
    // construction
    AxisymmetricSelfConsistent(df *DF, std::string MEfile,Potential_JS *exPot=nullptr)
        :df(ME,DF->ActionCalculator),DF(DF),
         ME(new MultipoleExpansion_Axisymmetric(data_folder+MEfile+".me")),
         DDD(Density_from_DF(DF)){
            if(exPot){
                ExternalPot = new MultiComponentPotential<Potential_JS>({ME,exPot});
                DF->reset_pot(ExternalPot);
            }
            else
                DF->reset_pot(ME);
            a0=ME->scaleradius();
            innerradius=ME->innerradius();
            outerradius=ME->outerradius();
    }

    inline double action_dist(const VecDoub& x){ return DF->action_dist(x);}
    inline double realspace_dist(const VecDoub& x){
        return DF->realspace_dist(x);}
    inline VecDoub realspace_dist_vec(const VecDoub& x){
        return DF->realspace_dist_vec(x);}

    // This code is used to build a model

    AxisymmetricSelfConsistent(df *DF, Potential_JS *InitPot, std::string Name, double a0 = 1., double irr=0.1, double orr=500., int nr=15, int na = 4, int lmax = 4, Potential_JS *ExternalPot=nullptr)
        :df(ME),DF(DF),DDD(Density_from_DF(DF)),InitPot(InitPot),
        name(data_folder+Name),a0(a0),innerradius(irr),outerradius(orr),ExternalPot(ExternalPot){

        std::cout<<"---------------------------------------------\n";
        std::cout<<"Constructing an axisymmetric self-consistent DF given by:\n"
                 <<DF->name()<<"\n\twith params "<<DF->params()
                 <<"\nwith an initial potential given by:\n"
                 <<InitPot->name()<<"\nwith params "<<InitPot->params()
                 <<"\nusing a multipole expansion with NR = "<<nr<<", NA = "
                 <<na<<", l_max = "<<lmax<<", a0 = "<<a0<<", ir = "<<irr
                 <<", or = "<<orr<<std::endl;
        if(ExternalPot)
            std::cout<<"and external potential: "<<ExternalPot->params()<<"\n";
        std::cout<<"---------------------------------------------\n";

        selfcon::writelogfile(DF,name);

        ME = new MultipoleExpansion_Axisymmetric(&DDD,
                                        nr,/* number of radial grid points */
                                        na,/* number of angular grid points */
                                        lmax, /* number of l modes */
                                        a0,/* scale radius */
                                        irr,orr);
        outerradius=ME->outerradius();
        // Output initial properties
        ME->visualize(name+".vis");
        ME->output_to_file(name+".me");
	selfcon::printMmaxtolog(name,0,ME->Mass(orr));
        selfcon::output_initial_pot(ME->get_radial_grid(),InitPot,name,"axisymmetric");
        if(ExternalPot)
            selfcon::output_ext_pot(ME->get_radial_grid(),ExternalPot,name,"axisymmetric");
    }

    void make_self_consistent(unsigned NITER, double gamma = 0.5);

};


class AxisymmetricMultiComSelfConsistent: public df{
    /* Calculates a self-consistent model */
private:

    const std::string data_folder = "data/";

    df_multicomponent *DFs;
    std::vector<MultipoleExpansion_Axisymmetric*> ME;
    std::vector<Density_from_DF> DDD;

    Potential_JS *InitPot;
    MultiComponentPotential<MultipoleExpansion_Axisymmetric> *TotalPot;

    std::string name;
    VecInt NR;
    VecDoub a0,innerradius, outerradius;
    const double threshold = 0.01; // quits when all potential devs less than

    Potential_JS *ExternalPot;

    VecDoub radial_grid;

public:

    void fill_radial_grid(void){
        double mini = Min(innerradius), maxo = Max(outerradius);
        double minstep = 1e9;
        for(unsigned i=0;i<DFs->ncompts();++i){
            double s = log(outerradius[i]/innerradius[i])/(double)NR[i];
            if(s<minstep)
                minstep=s;        }
        for(double ss=log(mini);ss<log(maxo);ss+=minstep)
            radial_grid.push_back(exp(ss));
    }

    // This code is used for finding the properties of the model after
    // construction
    AxisymmetricMultiComSelfConsistent(df_multicomponent *DFs, std::string MEfile,Potential_JS *exPot=nullptr)
        :df(TotalPot,DFs->ActionCalculator),DFs(DFs)
        {
            for(unsigned j=0;j<DFs->ncompts();++j){
                ME.push_back(new MultipoleExpansion_Axisymmetric(data_folder+MEfile+".me"+std::to_string(j)));
                DDD.push_back(Density_from_DF(DFs->compt(j)));
                a0.push_back((ME[j])->scaleradius());
                innerradius.push_back((ME[j])->innerradius());
                outerradius.push_back((ME[j])->outerradius());
            }

            TotalPot = new MultiComponentPotential<MultipoleExpansion_Axisymmetric>(ME);
            if(exPot){
                ExternalPot = new MultiComponentPotential<Potential_JS>({TotalPot,exPot});
                DFs->reset_pot(ExternalPot);
            }
            else
                DFs->reset_pot(TotalPot);
    }

    inline double action_dist(const VecDoub& x){ return DFs->action_dist(x);}
    inline double realspace_dist(const VecDoub& x){
        return DFs->realspace_dist(x);}
    inline VecDoub realspace_dist_vec(const VecDoub& x){
        return DFs->realspace_dist_vec(x);}

    // This code is used to build a model

    AxisymmetricMultiComSelfConsistent(df_multicomponent *DFs, Potential_JS *InitPot, std::string Name, VecDoub a0, VecDoub irr, VecDoub orr, VecInt nr, VecInt na, VecInt lmax, Potential_JS *ExternalPot=nullptr)
        :df(TotalPot),DFs(DFs),InitPot(InitPot)
        ,name(data_folder+Name),NR(nr),a0(a0),innerradius(irr),outerradius(orr),ExternalPot(ExternalPot){

        std::cout<<"---------------------------------------------\n";
        std::cout<<"Constructing a multicomponent axisymmetric self-consistent DF given by:\n"
                 <<DFs->name()<<"\nwith params "<<DFs->params()
                 <<"\nwith an initial potential given by:\n"
                 <<InitPot->name()<<"\nwith params "<<InitPot->params()
                 <<"\nusing a multipole expansions with ";
        for(unsigned i=0;i<DFs->ncompts();++i)
            std::cout<<std::to_string(i+1)+". NR = "<<nr[i]
                     <<", NA = "<<na[i]<<", l_max = "<<lmax[i]
                     <<", a0 = "<<a0[i]<<", ir = "<<irr[i]
                     <<", or = "<<orr[i]<<std::endl;
        if(ExternalPot)
            std::cout<<"and external potential: "<<ExternalPot->params()<<"\n";
        std::cout<<"---------------------------------------------\n";

        selfcon::writelogfile(DFs,name);

        fill_radial_grid();

        for(unsigned i=0;i<DFs->ncompts();++i){
            DDD.push_back(Density_from_DF(DFs->compt(i)));
            ME.push_back(new MultipoleExpansion_Axisymmetric(&(DDD[i]),
                            nr[i],/* number of radial grid points */
                            na[i],/* number of angular grid points */
                            lmax[i], /* number of l modes */
                            a0[i],/* scale radius */
                            irr[i],orr[i]));
            ME[i]->output_to_file(name+".me"+std::to_string(i));
            ME[i]->visualize(name+".vis"+std::to_string(i));
        }

        fill_radial_grid();

        TotalPot = new MultiComponentPotential<MultipoleExpansion_Axisymmetric>(ME);

        // Output initial properties
        selfcon::output_initial_pot(radial_grid,InitPot,name,"axisymmetric");
        if(ExternalPot)
            selfcon::output_ext_pot(radial_grid,ExternalPot,name,"axisymmetric");
    }

    void make_self_consistent(unsigned NITER, double gamma = 0.5);

};

// ============================================================================

class TriaxialSelfConsistent: public df{
    /* Calculates a self-consistent model */
private:

    const std::string data_folder = "data/";

    df *DF;
    MultipoleExpansion_Triaxial *ME;
    Density_from_DF DDD;
    Potential_JS *InitPot;
    std::string name;
    double a0;
    double innerradius, outerradius;
    const double threshold = 0.01; // quits when all potential devs less than
    Potential_JS *ExternalPot;

public:

    // This code is used for finding the properties of the model after
    // construction
    TriaxialSelfConsistent(df *DF, std::string MEfile,Potential_JS *exPot=nullptr)
        :df(ME,DF->ActionCalculator),DF(DF),
         ME(new MultipoleExpansion_Triaxial(data_folder+MEfile+".me")),
         DDD(Density_from_DF(DF)){
            pot=ME;
            if(exPot){
                ExternalPot = new MultiComponentPotential<Potential_JS>({ME,exPot});
                DF->reset_pot(ExternalPot);
            }
            else
                DF->reset_pot(ME);
            InitPot = ME;
            a0=ME->scaleradius();
            innerradius=ME->innerradius();
            outerradius=ME->outerradius();
    }

    inline double action_dist(const VecDoub& x){ return DF->action_dist(x);}
    inline double realspace_dist(const VecDoub& x){
        return DF->realspace_dist(x);}
    inline VecDoub realspace_dist_vec(const VecDoub& x){
        return DF->realspace_dist_vec(x);}
    // This code is used to build a model

    TriaxialSelfConsistent(df *DF, Potential_JS *InitPot, std::string Name, double a0 = 1., double irr=0.1, double orr=500., int nr=15, int na = 4, int lmax = 4,Potential_JS *ExternalPot=nullptr)
        :df(ME),DF(DF),DDD(Density_from_DF(DF)),InitPot(InitPot),
        name(data_folder+Name),a0(a0),innerradius(irr),outerradius(orr),ExternalPot(ExternalPot){

        std::cout<<"---------------------------------------------\n";
        std::cout<<"Constructing the triaxial self-consistent DF given by:\n"
                 <<DF->name()<<"\n\twith params "<<DF->params()
                 <<"\nwith an initial potential given by:\n"
                 <<InitPot->name()<<"\nwith params "<<InitPot->params()
                 <<"\nusing a multipole expansion with NR = "<<nr<<", NA = "
                 <<na<<", l_max = "<<lmax<<", a0 = "<<a0<<", ir = "<<irr
                 <<", or = "<<orr<<std::endl;
        if(ExternalPot)
            std::cout<<"and external potential: "<<ExternalPot->params()<<"\n";
        std::cout<<"---------------------------------------------\n";

        selfcon::writelogfile(DF,name);

        if(orr<static_cast<lmn_orb*>(DF->ActionCalculator)->get_ymax())
            std::cerr<< "Outer lmn_orb radius is greater than multipole grid"
                        "max radius. All orbits outside are Keplerian.\n";

        ME = new MultipoleExpansion_Triaxial(&DDD,
                                        nr,/* number of radial grid points */
                                        na,/* number of angular grid points */
                                        lmax, /* number of l modes */
                                        -1,/* no limit on m */
                                        a0,/* scale radius */
                                        irr,orr);
        outerradius=ME->outerradius();
        // Output initial properties
        ME->visualize(name+".vis");
        ME->output_to_file(name+".me");
	selfcon::printMmaxtolog(name,0,ME->Mass(orr));
        selfcon::output_initial_pot(ME->get_radial_grid(),InitPot,name,"triaxial");
        if(ExternalPot)
            selfcon::output_ext_pot(ME->get_radial_grid(),ExternalPot,name,"axisymmetric");
    }

    void make_self_consistent(unsigned NITER, double gamma = 0.5);
    VecDoub find_triaxiality(double r);
    void plot_triaxiality(void);
};

// ============================================================================
class TriaxialMultiComSelfConsistent: public df{
    /* Calculates a self-consistent model */
private:

    const std::string data_folder = "data/";

    df_multicomponent *DFs;
    std::vector<MultipoleExpansion_Triaxial*> ME;
    std::vector<Density_from_DF> DDD;

    Potential_JS *InitPot;
    MultiComponentPotential<MultipoleExpansion_Triaxial> *TotalPot;

    std::string name;
    VecInt NR;
    VecDoub a0,innerradius, outerradius;
    const double threshold = 0.01; // quits when all potential devs less than

    Potential_JS *ExternalPot;

    std::vector<double> radial_grid;

public:

    void fill_radial_grid(void){
        double mini = Min(innerradius), maxo = Max(outerradius);
        double minstep = 1e9;
        for(unsigned i=0;i<DFs->ncompts();++i){
            double s = log(outerradius[i]/innerradius[i])/(double)NR[i];
            if(s<minstep)
                minstep=s;
        }
        for(double ss=log(mini);ss<log(maxo);ss+=minstep)
            radial_grid.push_back(exp(ss));
    }

    // This code is used for finding the properties of the model after
    // construction
    TriaxialMultiComSelfConsistent(df_multicomponent *DFs, std::string MEfile,Potential_JS *exPot=nullptr)
        :df(TotalPot,DFs->ActionCalculator),DFs(DFs)
        {
            fill_radial_grid();
            for(unsigned j=0;j<DFs->ncompts();++j){
                ME.push_back(new MultipoleExpansion_Triaxial(data_folder+MEfile+".me"+std::to_string(j)));
                DDD.push_back(Density_from_DF(DFs->compt(j)));
                a0.push_back((ME[j])->scaleradius());
                innerradius.push_back((ME[j])->innerradius());
                outerradius.push_back((ME[j])->outerradius());
            }
            TotalPot = new MultiComponentPotential<MultipoleExpansion_Triaxial>(ME);
            if(exPot){
                ExternalPot = new MultiComponentPotential<Potential_JS>({TotalPot,exPot});
                DFs->reset_pot(ExternalPot);
            }
            else
                DFs->reset_pot(TotalPot);
    }

    inline double action_dist(const VecDoub& x){ return DFs->action_dist(x);}
    inline double realspace_dist(const VecDoub& x){
        return DFs->realspace_dist(x);}
    inline VecDoub realspace_dist_vec(const VecDoub& x){
        return DFs->realspace_dist_vec(x);}

    // This code is used to build a model

    TriaxialMultiComSelfConsistent(df_multicomponent *DFs, Potential_JS *InitPot, std::string Name, VecDoub a0, VecDoub irr, VecDoub orr, VecInt nr, VecInt na, VecInt lmax, Potential_JS *ExternalPot=nullptr)
        :df(TotalPot),DFs(DFs),InitPot(InitPot)
        ,name(data_folder+Name),NR(nr),a0(a0),innerradius(irr),outerradius(orr),ExternalPot(ExternalPot){

        std::cout<<"---------------------------------------------\n";
        std::cout<<"Constructing a multicomponent triaxial self-consistent DF given by:\n"
                 <<DFs->name()<<"\nwith params "<<DFs->params()
                 <<"\nwith an initial potential given by:\n"
                 <<InitPot->name()<<"\nwith params "<<InitPot->params()
                 <<"\nusing a multipole expansions with ";
        for(unsigned i=0;i<DFs->ncompts();++i)
            std::cout<<std::to_string(i+1)+". NR = "<<nr[i]
                     <<", NA = "<<na[i]<<", l_max = "<<lmax[i]
                     <<", a0 = "<<a0[i]<<", ir = "<<irr[i]
                     <<", or = "<<orr[i]<<std::endl;
        if(ExternalPot)
            std::cout<<"and external potential: "<<ExternalPot->params()<<"\n";
        std::cout<<"---------------------------------------------\n";

        selfcon::writelogfile(DFs,name);

        fill_radial_grid();

        for(auto o:orr){
            if(o<static_cast<lmn_orb*>(DFs->ActionCalculator)->get_ymax())
             std::cerr<< "Outer lmn_orb radius is greater than multipole grid"
                        "max radius. All orbits outside are Keplerian.\n";
        }
        for(unsigned i=0;i<DFs->ncompts();++i){
            DDD.push_back(Density_from_DF(DFs->compt(i)));
            ME.push_back(new MultipoleExpansion_Triaxial(&(DDD[i]),
                            nr[i],/* number of radial grid points */
                            na[i],/* number of angular grid points */
                            lmax[i], /* number of l modes */
                             -1,/* no limit on m */
                            a0[i],/* scale radius */
                            irr[i],orr[i]));
            ME[i]->output_to_file(name+".me"+std::to_string(i));
            ME[i]->visualize(name+".vis"+std::to_string(i));
        }

        fill_radial_grid();

        TotalPot = new MultiComponentPotential<MultipoleExpansion_Triaxial>(ME);

        // Output initial properties
        selfcon::output_initial_pot(radial_grid,InitPot,name,"triaxial");
        if(ExternalPot)
            selfcon::output_ext_pot(radial_grid,ExternalPot,name,"axisymmetric");
    }

    void make_self_consistent(unsigned NITER, double gamma = 0.5);

};
// ============================================================================

#endif
