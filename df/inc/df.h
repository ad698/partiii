#ifndef DF_H
#define DF_H
// ============================================================================

#include "utils.h"
#include "potential.h"
#include "lmn_orb.h"

// ============================================================================
// Action-based distribution function
// ============================================================================

class df{
public:

	inline virtual std::string name(void) const{
		return "You haven't written a description of this DF"; }
	inline virtual std::string params(void) const{ return "No param list"; }

	Potential_JS *pot;
	Action_Finder *ActionCalculator;
	double Mass, Norm, MN;
	std::string symmetry;

	df(Potential_JS *pot, Action_Finder *ActFinder=nullptr, std::string sym="spherical")
		:pot(pot), ActionCalculator(ActFinder), Mass(1.), Norm(1.), MN(Mass*Norm), symmetry(sym){}

	inline VecDoub PotGrad(const VecDoub& x){return pot->Forces(x);}
	double numerical_norm(double scale=1.);
	virtual double action_dist(const VecDoub& J) = 0;
	virtual double realspace_dist(const VecDoub& X){
		if(pot->H(X)>0.)return 0.;
		VecDoub J = ActionCalculator->actions(X);
		return action_dist(J);
	}
	virtual VecDoub realspace_dist_vec(const VecDoub& X){
		if(pot->H(X)>0.){return {0.,0.};}
		// for(auto i:X)std::cout<<i<<" ";
		VecDoub J = ActionCalculator->actions(X);
		// for(auto i:J)std::cout<<i<<" ";std::cout<<action_dist(J)<<std::endl;
		return {action_dist(J),J[3]};
	}

	void reset_pot(Potential_JS *Pot){
		pot=Pot;
		ActionCalculator->reset(Pot);
	}
	void reset_justpot(Potential_JS *Pot){
		pot=Pot;
	}
	void reset_pot(SphericalPotential *Pot){
		pot=Pot;
		ActionCalculator->reset_sph(Pot);
	}

	inline double get_mass(){return Mass;}
	inline void set_mass(double m){ Mass = m; MN= Mass*Norm;}

	double normalization(double IE = 1e-4, std::string type="Divonne");
	double normalization_action(double IE = 1e-4, std::string type="Divonne");
	double density(const VecDoub& x, double IE = 1e-4, std::string type="Divonne");
	double density_pol(const VecDoub& x, double IE = 1e-4, std::string type="Divonne");
	double projected_density(const VecDoub& x, const VecDoub& phi_theta, double size_in_zdirection=100.,double IE = 1e-4, std::string type="Divonne");
	double veldist(const VecDoub& x, int comp, double IE=1e-4);
	double density_cub(const VecDoub& x, double IE, int dd);
	void velocity_distributions(const VecDoub& x, const std::string& ofile, double IE);
	VecDoub moments(const VecDoub& x, double IE = 5e-4);
	VecDoub spherical_moments(double x);
	VecDoub xymoments(const VecDoub& x,double IE = 1e-3,int with_cross=1);
	VecDoub xzmoments(const VecDoub& x,double IE = 1e-3,int with_cross=1);
	VecDoub moments_mc(const VecDoub& x);
	VecDoub split_density(const VecDoub& x, double IE = 1e-4);
	// VecDoub split_density2(const VecDoub& x, double IE = 1e-4);
	void twoDdensity(const std::string& ofile);
	VecDoub testJeans(const VecDoub& x);
	void low_action_test(void);
	VecDoub ximoments(const VecDoub& x, int i, double IE = 1e-3);
	void xidisp(const std::string& ofile, double xmin = 1., double xmax = 20., double dx = 2.,double IE = 1e-3,int spherical=0);
	void xidisp(const std::string& ofile, double xmin = 1., double xmax = 20., int count = 10,std::string type="log",double IE = 1e-3,int spherical=0);
	void split_density(const std::string& ofile, double xmin, double xmax, int n, std::string type , double IE);
	void density(const std::string& ofile, double xmin, double xmax, int n, std::string type , double IE);
	void projected_density(const std::string& ofile, double xmax, int n,double IE);

	void sphr_disp(const std::string& ofile, double xmin = 1., double xmax = 20., double dx = 2.);
	void oneD_density(const std::string& ofile, double xmin, double xmax, int n, std::string type, double IE);
	void oneD_density(const std::string& ofile, VecDoub range, double IE);
};

class df_single_power: public df{
	private:
		const std::string desc =
		"Simple power law action-based DF,"
		"f(J) = (a_0|J_lambda|+a_1|J_mu|+a_2|J_nu|)^{-p}"
		"rho ~ r^-p";
		VecDoub a_i;
		double p;

	public:
		df_single_power(VecDoub a_i, double mass, double p, Potential_JS* Pot, Action_Finder *ActFinder)
			:df(Pot,ActFinder,"spherical"),a_i(a_i),p(p){Mass=mass;}

		inline std::string name(void) const{return desc;}
		inline std::string params(void) const{
			return "p = "+std::to_string(p)+
				 ", a_0 = "+std::to_string(a_i[0])+
				 ", a_1 = "+std::to_string(a_i[1])+
				 ", a_2 = "+std::to_string(a_i[2]);
		}

		double action_dist(const VecDoub& J);
};

class df_double_power: public df{
	private:
		const std::string desc =
		"Double power law action-based distribution function:\n\t"
		"f(J) = (J_0^s + (a_0|J_lambda|+a_1|J_mu|+a_2|J_nu|)^s)^{-p/s}*(J_c^s+(a_0|J_lambda|+a_1|J_mu|+a_2|J_nu|)^s)^{-q/s}.\n"
		"For a flat rotation curve the density at large radii falls off like"
		"\n\trho ~ r^-(p+q)"
		"\nand at small radii like"
		"\n\trho ~ r^-q";

		rand_gaussian *rnGauss;

		VecDoub a_i;
		double J0, p, q, s, Jc, Jfac;
		int use_fixed_ab;
		VecDoub ab;
		double noise;

	public:
		VecDoub av;
		bool printing;
		df_double_power(VecDoub a_i, double J0, double p, double q, double s,std::string symmetry, Potential_JS* Pot, Action_Finder *ActFinder, VecDoub ab = {}, double mass = 1., double Jc = 0.)
			:df(Pot,ActFinder),a_i(a_i),J0(J0),p(p),q(q),s(s),Jc(Jc)
			,Jfac(symmetry=="triaxial"?0.5:1.),ab(ab){

			rnGauss = new rand_gaussian(1.,1231312312);

			if(ab.size()>0)
				use_fixed_ab=1;
			else
				use_fixed_ab=0;
			if(Pot) pot = Pot;
			else pot = new NFW(10.,12.3e5,0.95,0.85);

			noise = -1.;
			av = VecDoub(5,0);

			if(p+q>3.) Norm = analytic_norm();
			else std::cerr<<"p+q<=3: norm is infinite\n";
			Mass = mass;

			printing = false;
		}
		inline std::string name(void) const{return desc;}
		inline std::string params(void) const{
		  return "p = "+std::to_string(p)+", q = "+std::to_string(q)+
		  		 ", s = "+std::to_string(s)+", J_0="+std::to_string(J0)+
		  		 ", a_0 = "+std::to_string(a_i[0])+
				 ", a_1 = "+std::to_string(a_i[1])+
				 ", a_2 = "+std::to_string(a_i[2])+
				 ", J_c = "+std::to_string(Jc)+
				 ", sym = "+symmetry;
		}

		void scale_central_density_to_unity(void){
			double d = density(VecDoub(3,0.001));
			Mass/=d;
		}

		double analytic_norm(void){
			// for q=0
			if(p+q>3.){
				if(q!=0.) return numerical_norm(J0);
				return pow(2.*PI,3)/a_i[0]/a_i[1]/a_i[2]*pow(J0,-p+3)/(p-1)/(p-2)/(p-3);
			}
			else { std::cerr<<"p<3: norm is infinite\n"; return 0.;}
		}
		void set_noise(double n){noise=n;}
		void set_ab(const VecDoub& AB){ab=AB;}
		double action_dist(const VecDoub& J);
		double realspace_dist(const VecDoub& J);
		VecDoub realspace_dist_vec(const VecDoub& J);
};

class df_we: public df{
	private:
		const std::string desc =
		"Action-based distribution function from Williams & Evans (2015):\n\t"
		"f(J) = T(J)*L(J)^{-lambda}/(J_0^2+L(J)^2)^((mu-lambda)/2)";
		double J0, lambda, mu, D0, D1, Sa, Sg, Jb, qz, b0, Jfac;

	public:
		df_we(double J0, double lambda, double mu, double D0, double D1, double Sa, double Sg, double Jb, Potential_JS* Pot, Action_Finder *ActFinder, double mass = 1., double Jc = 0.,double qz=1.,double b0=0.,std::string symmetry="spherical")
			:df(Pot,ActFinder,symmetry),J0(J0),lambda(lambda),mu(mu),D0(D0),D1(D1),Sa(Sa),Sg(Sg),Jb(Jb),qz(qz),b0(b0)
				,Jfac(symmetry=="triaxial"?0.5:1.){
				Norm=1./numerical_norm(J0);
				Mass=mass;
				MN=Mass*Norm;
			}

		inline std::string name(void) const{return desc;}
		inline std::string params(void) const{
		  return "J_0 = "+std::to_string(J0)+
				 ", lambda = "+std::to_string(lambda)+
				 ", mu = "+std::to_string(mu)+
				 ", D0 = "+std::to_string(D0)+
				 ", D1 = "+std::to_string(D1)+
				 ", Sa = "+std::to_string(Sa)+
				 ", Sg = "+std::to_string(Sg)+
				 ", Jb = "+std::to_string(Jb)+
				 ", b0 = "+std::to_string(b0)+
				 ", qz = "+std::to_string(qz)+
				 ", sym = "+symmetry+
				 ", M = "+std::to_string(Mass);
		}
		VecDoub get_params(void){
			VecDoub params(11,0.);
			params[0]=J0;	params[1]=lambda;	params[2]=mu;
			params[3]=D0;	params[4]=D1;		params[5]=Sa;
			params[6]=Sg;	params[7]=Jb;		params[8]=qz;
			params[9]=b0;	params[10]=Mass;
			return params;
		}
		void set_params(const VecDoub& params){
			J0=params[0];	lambda=params[1];	mu=params[2];
			D0=params[3];	D1=params[4];		Sa=params[5];
			Sg=params[6];	Jb=params[7];		qz=params[8];
			b0=params[9];
			MN=1.;
			Norm = 1./numerical_norm(J0);
			Mass = params[10];
			MN= Mass*Norm;
		}
		double action_dist(const VecDoub& J);
};

class df_pb: public df{
	private:
		const std::string desc =
		"Action-based distribution function from Posti et al. (2015):\n\t"
		"f(J) = (1+J_0/h(J))^{(6-alpha)/(4-alpha)}/(1+g(J)/J_0)^(2*beta-3)";
		double J0, alpha, beta, Jfac;

	public:
		df_pb(double J0, double alpha, double beta,std::string symmetry, Potential_JS* Pot, Action_Finder *ActFinder, double mass = 1., double Jc = 0.)
			:df(Pot,ActFinder,symmetry),J0(J0),alpha(alpha),beta(beta)
			,Jfac(symmetry=="triaxial"?0.5:1.){
				Norm = 1./numerical_norm(J0);
				Mass = mass;
				MN = Mass*Norm;
			}

		inline std::string name(void) const{return desc;}
		inline std::string params(void) const{
		  return "J_0 = "+std::to_string(J0)+
				 ", alpha = "+std::to_string(alpha)+
				 ", beta = "+std::to_string(beta)+
				 ", sym = "+symmetry;
		}

		virtual double action_dist(const VecDoub& J);
};

class df_isochrone: public df{
	// Simple action-based distribution function
	// Isochrone
	private:
		VecDoub a_i;
		double GM, b, Jfac, Norm;

		const std::string desc =
		"Isotropic isochrone action-based distribution function:\n\t"
		"f(J) = f(H(J)).\n";

	public:
		df_isochrone(VecDoub a_i, double GM, double b, std::string symmetry, Potential_JS* Pot, Action_Finder *ActFinder)
			:df(Pot,ActFinder,symmetry),a_i(a_i), GM(GM), b(b),Jfac(symmetry=="triaxial"?0.5:1.),Norm(1.){
				Norm=1./numerical_norm(sqrt(GM*b))/GM;
				Mass=GM/conv::G;
				MN=Mass*Norm;
		}
		inline std::string name(void) const{return desc;}
		inline std::string params(void) const{
		  return "alpha_phi = "+std::to_string(a_i[0])+
		  		 ", alpha_z = "+std::to_string(a_i[1])+
				 ", M="+std::to_string(GM)+", b = "+std::to_string(b)
				 +", sym = "+symmetry;
		}
		// virtual ~df_isochrone(){if(!passed)delete lmn;}
		double action_dist(const VecDoub& J);
};

class df_isochrone_spherical: public df{
	// Simple action-based distribution function
	// Isochrone
	private:
		double GM, b;
	public:
		df_isochrone_spherical(double GM, double b, Potential_JS* Pot)
			:df(Pot),GM(GM), b(b){
				Mass = GM/conv::G; MN = Mass*Norm;
		}
		double action_dist(const VecDoub& J);
		double realspace_dist(const VecDoub& J);
		VecDoub realspace_dist_vec(const VecDoub& J);
};

class df_multicomponent: public df{
	private:
		std::vector<df*> df_array;
		const std::string desc =
		"Multi-component action-based DF f(J) = \\sum_i f_i(J):\n";
	public:
		df_multicomponent(Potential_JS *Pot, Action_Finder *Act):df(Pot,Act){}

		void add_new_df(df *DF){df_array.push_back(DF);}
		unsigned ncompts(void){return df_array.size();}
		df* compt(int i) const{return df_array[i];}

		inline std::string name(void) const{
			std::string n = desc;int j=1;
			for(auto i:df_array){
				n+=std::to_string(j)+": "+i->name()+"\n";
				++j;
			}
			return n;
		}
		inline std::string params(void) const{
		  	std::string n;int j=1;
			for(auto i:df_array){
				n+=std::to_string(j)+": "+i->params()+"\n";
				++j;
			}
			return n;
		}

		double action_dist(const VecDoub &J){
			double sum = 0.;
			for(auto i:df_array) sum+=i->action_dist(J);
			return sum;
		}
};


struct density_st{
	df *DF;
	VecDoub x;
	VecDoub x2min,x2max;
	int pp;
	double scale;
	density_st(df *DF, VecDoub x, VecDoub x2min, VecDoub x2max, int pp, double scale=1.)
		:DF(DF),x(x),x2min(x2min),x2max(x2max),pp(pp),scale(scale){}
};

struct projected_density_st: density_st{
	// defines Cartesian coordinates in the observation plane
	// defines radial vector = line of sight -- note there is no rotation about l.o.s
	VecDoub phi_theta, n, phihat, thetahat;

	projected_density_st(df *DF, VecDoub x, VecDoub phi_theta, VecDoub x2min, VecDoub x2max, int pp)
		:density_st(DF,x,x2min,x2max,pp),phi_theta(phi_theta){
			double cp = cos(phi_theta[0]), sp = sin(phi_theta[0]);
			double ct = cos(phi_theta[1]), st = sin(phi_theta[1]);
			n = {st*cp,st*sp,ct};
			phihat={-sp,cp,0.};
			thetahat={-cp*ct,-ct*sp,st};
		}
};

struct veldist_st : density_st{
	int swit;
	veldist_st(df *DF, VecDoub x, VecDoub x2min, VecDoub x2max, int swit)
		:density_st(DF,x,x2min,x2max,-1),swit(swit){}
};

#endif
// ============================================================================
