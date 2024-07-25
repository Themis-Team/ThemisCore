// Include header
#include <nullgeodesic.h>
#ifdef MPI_MAP
#include "mpi.h"
#endif

namespace VRT2 {

//#define CAUSALLY_PEDANTIC // Does nothing right now.
#define DYNAMICAL_DRMAX

NullGeodesic::NullGeodesic(Metric& metric, RadiativeTransfer& rt, StopCondition& stop)
  : _g(metric), _x0(metric), _k0(metric), _disp(metric), _rt(rt), _stop(stop)
{
  _iquv.resize(4,0.0);
}

NullGeodesic::NullGeodesic(double y[], Metric& metric, RadiativeTransfer& rt, StopCondition& stop)
  : _g(metric), _x0(metric), _k0(metric), _disp(metric), _rt(rt), _stop(stop)
{
  _iquv.resize(4,0.0);
  _g.reset(y);
  _x0.mkcon(y);
  _k0.mkcov(y+4);
  reinitialize(_x0,_k0);
}

NullGeodesic::NullGeodesic(FourVector<double>& x0,FourVector<double>& k0,
			   Metric& metric, RadiativeTransfer& rt, StopCondition& stop)
  : _g(metric), _x0(metric), _k0(metric), _disp(metric), _rt(rt), _stop(stop)
{
  _iquv.resize(4,0.0);
  reinitialize(x0,k0);
}

/*** Reset position (must also reset disp, pol, and rad) ***/
void NullGeodesic::reinitialize(FourVector<double>& x0,FourVector<double>& k0)
{
  _x0 = x0;
  _k0 = k0;

  // Reset metric
  _g.reset(_x0.con());

  // Normalize _k0 and tell _rt
  _rt.set_frequency_scale(std::fabs(_k0.cov(0)));
  _stop.set_frequency_scale(std::fabs(_k0.cov(0)));
  _k0 *= 1.0/std::fabs(_k0.cov(0));

  // Reset stuff
  _disp.reinitialize(_x0,_k0); // Dispersion relation
  _rt.reinitialize(_x0,_k0);   // Radiative Transfer
  // Reset _y_ray
  for (int i1=0;i1<=NDIM_NAR;++i1){
    _y_ray[i1].resize(0);
    _dydx_ray[i1].resize(0);
  }
}

/*** Return Integrated Values ***/
std::valarray<double> NullGeodesic::IQUV() { return ( _iquv ); }
double NullGeodesic::tau() { return ( _tau_int ); }
double NullGeodesic::D() { return ( _D_int ); }

/*** Output ray (polar v. cartesian) ***/
void NullGeodesic::output_ray(std::string fname,int coord)
{
  // open file
  std::ofstream ray_out(fname.c_str());
  output_ray(ray_out,coord);
}
void NullGeodesic::output_ray(std::ostream& ray_out, int coord)
{
  ray_out.setf(std::ios::scientific);
  ray_out << std::setprecision(3);

  double y_local[NDIM_NAR];

  for (int i=1;i<NDIM_NAR;++i)
    y_local[i] = _y_ray[i][0];

  // Set _y_save;
  save_y( y_local ,1);

  if (coord==0){ // polar
    // Print headers
    ray_out << "Metric values: " << _g.ang_mom();
    ray_out << std::setw(12) << "lambda"
	    << std::setw(12) << "t"
	    << std::setw(12) << "r"
	    << std::setw(12) << "theta"
	    << std::setw(12) << "phi"
	    << std::setw(12) << "k_t"
	    << std::setw(12) << "k_r"
	    << std::setw(12) << "k_th"
	    << std::setw(12) << "k_ph"
	    << std::setw(12) << "D_avg" << std::endl;
    for(int i1=0;i1<int(_y_ray[0].size());++i1){
      // reset metric
      _g.reset(_y_ray[1][i1],_y_ray[2][i1],_y_ray[3][i1],_y_ray[4][i1]);
      for (int i=1;i<NDIM_NAR;++i)
	y_local[i] = _y_ray[i][i1];
      if (save_y(y_local,0) || !i1 || i1==(int(_y_ray[0].size())-1)){
	ray_out << std::setw(12) << _y_ray[0][i1];
	for(int i2=1;i2<NDIM_NAR;++i2)
	  ray_out << std::setw(12) << _y_ray[i2][i1];
	ray_out << std::endl;
      }
    }
  }
  else if (coord==1){
    // Print headers
    ray_out << std::setw(12) << "lambda"
	    << std::setw(12) << "t"
	    << std::setw(12) << "x"
	    << std::setw(12) << "y"
	    << std::setw(12) << "z"
	    << std::setw(12) << "k^t"
	    << std::setw(12) << "k^x"
	    << std::setw(12) << "k^y"
	    << std::setw(12) << "k^z"
	    << std::setw(12) << "D_avg" << std::endl;
    double r, sn_th, cs_th, sn_ph, cs_ph;
    double x, y, z, k_x, k_y, k_z;
    for(int i1=0;i1<int(_y_ray[0].size());++i1){
      // reset metric
      _g.reset(_y_ray[1][i1],_y_ray[2][i1],_y_ray[3][i1],_y_ray[4][i1]);
      for (int i=1;i<NDIM_NAR;++i)
	y_local[i] = _y_ray[i][i1];
      if (save_y(y_local,0) || !i1 || i1==(int(_y_ray[0].size())-1)){
	// intermediate defs
	r = _y_ray[2][i1];
	sn_th = std::sin(_y_ray[3][i1]);
	cs_th = std::cos(_y_ray[3][i1]);
	sn_ph = std::sin(_y_ray[4][i1]);
	cs_ph = std::cos(_y_ray[4][i1]);
	// x,y,z
	x = r*sn_th*cs_ph;
	y = r*sn_th*sn_ph;
	z = r*cs_th;
	FourVector<double> k(_g);
	k.mkcov(_y_ray[5][i1],_y_ray[6][i1],_y_ray[7][i1],_y_ray[8][i1]);
	// k^x, k^y, k^z
	k_x = k.con(1)*sn_th*cs_ph + k.con(2)*r*cs_th*cs_ph - k.con(3)*r*sn_th*sn_ph;
	k_y = k.con(1)*sn_th*sn_ph + k.con(2)*r*cs_th*sn_ph + k.con(3)*r*sn_th*cs_ph;
	k_z = k.con(1)*cs_th - k.con(2)*r*sn_th;
	// Output to file
	ray_out << std::setw(12) << _y_ray[0][i1]
		<< std::setw(12) << _y_ray[1][i1]
		<< std::setw(12) << x
		<< std::setw(12) << y
		<< std::setw(12) << z
		<< std::setw(12) << k.con(0)
		<< std::setw(12) << k_x
		<< std::setw(12) << k_y
		<< std::setw(12) << k_z
		<< std::setw(12) << _y_ray[9][i1]
		<< std::endl;
      }
    }
  }
  else
    std::cerr << "In NullGeodesic, sorry, coord=" << coord 
	 << " is not an option" << std::endl;
}


/*** Derivatives for rkqs and rkck in propagate ***/
void NullGeodesic::derivs(double x, double y[], double dydx[])
{
  // Declarations
  double reparam;

  // Reset stuff
  _g.reset(y);
  _disp.reinitialize(y);
  _rt.reinitialize(y);

  // Get reparametrization to make things okay at horizon
  reparam = reparametrize(y);

  // Assign dydx
  for (int i=0; i<4; ++i){
    dydx[i]   = reparam * _disp.dD_dk(0).con(i);   // dx/dlambda
    dydx[i+4] = - reparam * _disp.dD_dx(0).cov(i); // dk/dlambda

    if (vrt2_isnan(dydx[i]))
      dydx[i] = 1.0e10;
    if (vrt2_isnan(dydx[i+4]))
      dydx[i+4] = 1.0e10;
  }
  // D -> for check  
  dydx[8] = reparam * _disp.D(0);
  if (vrt2_isnan(dydx[8]))
    dydx[8] = 1.0e10;
  // tau -> for check (reparam is already in dydx)
  dydx[9] = _rt.isotropic_absorptivity(dydx);
  if (vrt2_isnan(dydx[9]))
    dydx[9] = 1.0e10;
}

/*** Reparametrization to Remove Singular behaviour near Horizon ***/
double NullGeodesic::reparametrize(double y[])
{
  if (1.0/_g.ginv(0,0)<0.0) 
    return ( y[1]*y[1]*std::pow(std::min(1.0-_g.horizon()/y[1],std::fabs(1.0/_g.ginv(0,0))),1.0) );
  else
    return ( 1.0E300 );

  /*
  double rhoriz = _g.horizon();
  if (y[1] > rhoriz) 
    //return ( y[1]*y[1]/std::sqrt(std::fabs(_g.ginv(0,0))) );
    return ( y[1]*y[1]*std::pow(1.0-rhoriz/y[1],0.5) );
  else
    return ( 1.0E300 );
  */
}

/*** Relative error scalings ***/
#define TINY 1.0e-2 // VERY Sensitive -> Controls time!
void NullGeodesic::get_yscal(double h, double x, double y[], double dydx[], double yscal[])
{
  yscal[0] = std::fabs(y[0])+1.0*std::fabs(dydx[0]*h)+TINY; // t
  yscal[1] = std::fabs(y[1])+1.0*std::fabs(dydx[1]*h)+TINY; // r
  yscal[2] = VRT2_Constants::pi; // theta
  yscal[3] = VRT2_Constants::pi; // phi
  yscal[4] = std::fabs(y[4]) + TINY; // k_t
  yscal[5] = std::fabs(y[4]) + TINY; // k_r
  yscal[6] = y[1]*std::fabs(y[4]) + TINY; // k_theta
  yscal[7] = y[1]*std::sin(y[2])*std::fabs(y[4]) + TINY; // k_phi
  yscal[8] = 0.0*std::fabs(y[8]) + 1.0*std::fabs(dydx[8]*h) + TINY; // D
  yscal[9] = 0.0*std::fabs(y[9]) + 1.0*std::fabs(dydx[9]*h) + TINY; // tau
}
#undef TINY

#define MAXSTP 200000
/*** Propagate ray ***/
// Returns string of filenames
std::vector<std::string> NullGeodesic::propagate(double h, std::string output)
{
  int out_coords = 1;

  /** Set maximum dr for a step **/
#ifdef DYNAMICAL_DRMAX
  double dr_max_power = 1;
  double dr_max = 4.0;
#else
  double dr_max = 4.0;
#endif

  /** Zero dispersion relation **/
  FourVector<double> k = _disp.Zero_D(0);
  _disp.reinitialize(_x0,k);

  if (k.cov(0)==0.0 || std::fabs(k*k)>10) {
    std::cerr << "Failed to zero dispersion relation in NullGeodesic\n";
    std::cerr << "k2 = " <<  (k*k) << "  k02 = " << (_k0*_k0) << '\n';
    k = _k0;
  }

  /** Set y for integration (the last two are I, tau, and D(for check)) **/
  double y[NDIM_NAR] = {_x0.con(0),_x0.con(1),_x0.con(2),_x0.con(3), // x
			k.cov(0),k.cov(1),k.cov(2),k.cov(3),         // k
			0.0,0.0};                                    // D, tau
  

  /** Integrate **/
  /* Preliminaries */
  // Define x, yscal, and dydx
  double x=0.0, yscal[NDIM_NAR], dydx[NDIM_NAR];
  // Define saved quantities
  double x0, y0[NDIM_NAR];
  // Define rkqs quantities
  double eps = 1E-7;
  double hdid=0, hnext=0, hstart=0;
  double hmax=2.0E-2;
  // Define stop
  unsigned int stop;
  unsigned int going_to_adi;
  bool not_stepping = true;
#ifdef CAUSALLY_PEDANTIC
  // Define FourVector for causal check
  FourVector<double> ktmp(_g);
#endif
  double dl2_max=1;
  std::vector<std::string> outnames(1);
  std::string fname;
  if (output[0]!='!')
    fname = output;

  // Get derivatives for first rkqs
  derivs(x,y,dydx);

  /* Select initial stepsize (Note hmax is to big for kerr at start!) */
  hstart = 1.0E-4*hmax;
#ifdef DYNAMICAL_DRMAX
  double dldx_space = 0.0;
  for (int i=1; i<4; ++i)
    for (int j=1; j<4; ++j)
      dldx_space += dydx[i]*dydx[j] * _g.g(i,j);
  dldx_space = std::sqrt(dldx_space);
  h = std::min(hstart,std::fabs(dr_max*std::pow(y[1]/200.0,dr_max_power)/dldx_space)); // In comparison to Delta r <= dr_max;
#else
  h = std::min(hstart,std::fabs(dr_max/dydx[1])); // In comparison to Delta r <= dr_max;
#endif

  // Add first to y_ray and dydx_ray
  add_to_y_ray(x,y,dydx);

  /* Begin Loop */
  for (int nstp=1; nstp<=MAXSTP; nstp++){
    // Get yscal for rkqs
    get_yscal(h,x,y,dydx,yscal);

    // Save step if it is neccessary to step up to stop
    x0 = x;
    for (int i=0; i<NDIM_NAR; i++)
      y0[i] = y[i];

    bool bollocks = false;


    do{
      // rkqs step
      rkqs(y,dydx,NDIM_NAR,x,-h,eps,yscal,hdid,hnext);
      /*
      if (rkqs(y,dydx,NDIM_NAR,x,-h,eps,yscal,hdid,hnext)) {
	std::cerr << "Stepsize driven to zero in NullGeodesic::propagate\n";
#ifdef MPI_MAP
	std::cerr << "rank = " << MPI::COMM_WORLD.Get_rank() << '\n';
#endif
	std::cerr << "(r,theta) = (" << y[1] << ',' << y[2] << ")\n";
	std::cerr << "hdid = " << hdid << '\n';
	std::cerr << "g^tt = " << _g.ginv(0,0)
		  << " g^rr = " << _g.ginv(1,1)
		  << " g^yy = " << _g.ginv(2,2)
		  << " g^ff = " << _g.ginv(3,3)
		  << " g^tf = " << _g.ginv(0,3)
		  << '\n'
		  << "Dg^tt_,r = " << _g.Dginv(0,0,1)
		  << " Dg^tt_,y = " << _g.Dginv(0,0,2)
		  << " Dg^rr_,r = " << _g.Dginv(1,1,1)
		  << " Dg^rr_,y = " << _g.Dginv(1,1,2)
		  << " Dg^yy_,r = " << _g.Dginv(2,2,1)
		  << " Dg^yy_,y = " << _g.Dginv(2,2,2)
		  << " Dg^ff_,r = " << _g.Dginv(3,3,1)
		  << " Dg^ff_,f = " << _g.Dginv(3,3,2)
		  << '\n'
		  << "dD/dt = " << _disp.dD_dx(0).cov(0)
		  << " dD/dr = " << _disp.dD_dx(0).cov(1)
		  << " dD/dy = " << _disp.dD_dx(0).cov(2)
		  << " dD/df = " << _disp.dD_dx(0).cov(3)
		  << '\n';
	_g.reset(y);
	FourVector<double> ktmp(_g);
	ktmp.mkcov(y+4);
	std::cerr << "k2 = " << (ktmp*ktmp) << '\n';
	ktmp.mkcov(y0+4);
	std::cerr << "k02 = " << (ktmp*ktmp) << '\n';
	for (int i=0; i<NDIM_NAR; i++)
	  std::cerr << std::setw(15) << y[i]
		    << std::setw(15) << dydx[i]
		    << std::setw(15) << yscal[i]
		    << std::setw(15) << y0[i]
		    << '\n';
	bollocks = true;
	//std::abort();
      }
      */
	  
      hdid *= -1.0; // keep h positive for legacy issues
      hnext *= -1.0;

      // Get derivatives for next rkqs
      derivs(x,y,dydx);

      /*
      if (MPI::COMM_WORLD.Get_rank()==0)
      {
	FourVector<double> ktmp(_g);
	ktmp.mkcov(y+4);
	std::cout << std::setw(15) << y[0]
		  << std::setw(15) << y[1]
		  << std::setw(15) << y[2]
		  << std::setw(15) << (ktmp*ktmp)
		  << std::endl;
      }
      */
      // Set new stepsize if not stepping to stop
      if (not_stepping) {
	h = std::min(hnext,hmax); // In comparison to hmax
#ifdef DYNAMICAL_DRMAX
	dldx_space = 0.0;
	for (int i=1; i<4; ++i)
	  for (int j=1; j<4; ++j)
	    dldx_space += dydx[i]*dydx[j] * _g.g(i,j);
	dldx_space = std::sqrt(dldx_space);
	h = std::min(h,std::fabs(dr_max*std::pow(y[1]/200.0,dr_max_power)/dldx_space)); // In comparison to Delta r <= dr_max
#else
	h = std::min(h,std::fabs(dr_max/dydx[1])); // In comparison to Delta r <= dr_max
#endif
      }
      
      // Get cut off conditions
      _g.reset(y);
      _disp.reinitialize(y);
      stop = _stop.stop_condition(y,dydx) || y[9]<-10.0 || bollocks;  // or if tau>10 or stalled
      dl2_max = _rt.dlambda(y,dydx);

      going_to_adi = stop;
      if (going_to_adi) {
	if (not_stepping)
	  not_stepping = false;
	// Get distance change for lower limit
	going_to_adi = going_to_adi && ( std::fabs(x-x0) > 1.0e-6/dl2_max ); 
      }
      if (going_to_adi) {
	// Set back to last step
	x=x0;
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
	for (int i=1; i<NDIM_NAR; i++)
	  y[i] = y0[i];
	// Decrease step size
	h = 0.1*hdid;
	// Get derivatives for next rkqs
	derivs(x,y,dydx);
      }
    } while(going_to_adi);

    // Add it to y_ray and dydx_ray
    add_to_y_ray(x,y,dydx);

    // Check stop condition (return if done)
    if (stop) {

      // Save current integrals
      _D_int = y[8];
      _tau_int = -y[9]; 

      // Set initial intensities to zero
      _iquv = _stop.IQUV(y);

      // Integrate up iquv
      _iquv = _rt.IQUV_integrate(_y_ray,_dydx_ray,_iquv);
 
      // Check to see if to print and return
      if (output[0]!='!'){
	      output_ray(fname+"n.d",out_coords);
	      outnames.resize(0);
	      outnames.push_back(fname+"n.d");
      }
      else{
	outnames[0] = ("null");
      }

      return (outnames);
    }
  }

  // If it hasn't returned yet, there were too many steps
  std::cerr << "Max steps were reached in nullgeodesic propagation!\n";
  std::cerr << _g.ang_mom() << std::endl;
  // Check to see if to print and return
  if (output[0]=='!'){
    output_ray(fname+"n.d",out_coords);
    outnames.resize(0);
    outnames.push_back(fname+"n.d");
  }
  else{
    outnames[0] = ("null");
  }
  return (outnames);
}
#undef CYC_CUT
#undef MAXSTP


/*** Check to see if to save y ***/
int NullGeodesic::save_y(double y[], int set)
{
  //return 1;
  int N_save = NDIM_NAR/2;

  if (set)
    for (int i1=0;i1<N_save;++i1)
      _y_save[i1] = y[i1];
  else{
    double dl2=0;
    for (int i1=0;i1<_g.Ng;i1++){
      if (_g.gi[i1]*_g.gj[i1])
	dl2 += (y[_g.gi[i1]]-_y_save[_g.gi[i1]])
	  * (y[_g.gj[i1]]-_y_save[_g.gj[i1]])
	  * _g.g(i1);
    }
    if (std::sqrt(dl2)>=0.1){
      for (int i1=0;i1<N_save;++i1)
	_y_save[i1] = y[i1];
      return 1;
    }
  }
  return 0;
}

/*** Add members to y_ray ***/
void NullGeodesic::add_to_y_ray(double x, double y[], double dydx[])
{
  _y_ray[0].push_back(x);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (int i=0; i<NDIM_NAR; ++i){
    _y_ray[i+1].push_back(y[i]);
    _dydx_ray[i+1].push_back(dydx[i]);
  }
  _y_ray[NDIM_NAR-1].push_back(_disp.D(0));
}
/*** Backup in y_ray to first position before lambda = x ***/
void NullGeodesic::backup_y_ray(double x)
{
  for (;_y_ray[0][_y_ray[0].size()-1]>=x && _y_ray[0].size()>0;) {
    _y_ray[0].pop_back();
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
    for (int i=0; i<NDIM_NAR; ++i) {   
      _y_ray[i+1].pop_back();
      _dydx_ray[i+1].pop_back();
    }
    _y_ray[9].pop_back();
  }
}

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
int NullGeodesic::rkqs(double y[], double dydx[], int n, double& x, double htry, 
			  double eps, double yscal[], double& hdid, double& hnext)
{
  static int i;
  static double errmax,h,htemp,xnew,yerr[NDIM_NAR],ytemp[NDIM_NAR]; 
  
  h=htry;
  for (;;) {
    rkck(y,dydx,n,x,h,ytemp,yerr);
    errmax=0.0;
    for (i=0;i<n;i++){
      if (std::fabs(yerr[i]/yscal[i])>errmax)
	errmax = std::fabs(yerr[i]/yscal[i]);
      if (vrt2_isnan(ytemp[i]))
	errmax = 10.0*eps;
    }
    errmax /= eps;

    if (errmax <= 1.0)
      break;

    htemp = ( vrt2_isinf(errmax) ? 0.5*h : SAFETY*h*std::pow(errmax,PSHRNK) );
    htemp = (h >= 0.0 ? std::max(htemp,0.1*h) : std::min(htemp,0.1*h));
    xnew=x+htemp;

    if (xnew == x){
      return 1;
    }
    h = htemp;
  }

  if (errmax > ERRCON)
    hnext=SAFETY*h*std::pow(errmax,PGROW);
  else
    hnext=5.0*h;
  x += (hdid=h);

  for (i=0;i<n;i++)
    y[i]=ytemp[i];

  return 0;
}
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON

void NullGeodesic::rkck(double y[], double dydx[], int n, double x, double h, 
			   double yout[], double yerr[])
{
  int i;
  static constexpr double a2=0.2,a3=0.3,a4=0.6,a5=1.0,a6=0.875,b21=0.2,
    b31=3.0/40.0,b32=9.0/40.0,b41=0.3,b42 = -0.9,b43=1.2,
    b51 = -11.0/54.0, b52=2.5,b53 = -70.0/27.0,b54=35.0/27.0,
    b61=1631.0/55296.0,b62=175.0/512.0,b63=575.0/13824.0,
    b64=44275.0/110592.0,b65=253.0/4096.0,c1=37.0/378.0,
    c3=250.0/621.0,c4=125.0/594.0,c6=512.0/1771.0,
    dc5 = -277.00/14336.0;
  static constexpr double dc1=c1-2825.0/27648.0,dc3=c3-18575.0/48384.0,
    dc4=c4-13525.0/55296.0,dc6=c6-0.25;
  static double ak2[NDIM_NAR],ak3[NDIM_NAR],ak4[NDIM_NAR],
    ak5[NDIM_NAR],ak6[NDIM_NAR],ytemp[NDIM_NAR];

#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (i=0;i<NDIM_NAR;i++)
    ytemp[i]=y[i]+b21*h*dydx[i];
  derivs(x+a2*h,ytemp,ak2);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (i=0;i<NDIM_NAR;i++)
    ytemp[i]=y[i]+h*(b31*dydx[i]+b32*ak2[i]);
  derivs(x+a3*h,ytemp,ak3);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (i=0;i<NDIM_NAR;i++)
    ytemp[i]=y[i]+h*(b41*dydx[i]+b42*ak2[i]+b43*ak3[i]);
  derivs(x+a4*h,ytemp,ak4);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (i=0;i<NDIM_NAR;i++)
    ytemp[i]=y[i]+h*(b51*dydx[i]+b52*ak2[i]+b53*ak3[i]+b54*ak4[i]);
  derivs(x+a5*h,ytemp,ak5);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (i=0;i<NDIM_NAR;i++)
    ytemp[i]=y[i]+h*(b61*dydx[i]+b62*ak2[i]+b63*ak3[i]+b64*ak4[i]+b65*ak5[i]);
  derivs(x+a6*h,ytemp,ak6);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (i=0;i<NDIM_NAR;i++)
    yout[i]=y[i]+h*(c1*dydx[i]+c3*ak3[i]+c4*ak4[i]+c6*ak6[i]);
#if defined(OPENMP_PRAGMA_VECTORIZE)
#pragma omp simd
#endif
  for (i=0;i<NDIM_NAR;i++)
    yerr[i]=h*(dc1*dydx[i]+dc3*ak3[i]+dc4*ak4[i]+dc5*ak5[i]+dc6*ak6[i]);
}

#ifdef CAUSALLY_PEDANTIC
#undef CAUSALLY_PEDANTIC
#endif

#ifdef DYNAMICAL_DRMAX
#undef DYNAMICAL_DRMAX
#endif
};
