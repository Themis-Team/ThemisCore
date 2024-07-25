// Include Statements
#include "dispersionrelation.h"

/*** CONSTRUCTORS ***/
// Don't initialize
namespace VRT2 {
DispersionRelation::DispersionRelation(Metric& metric) 
  : _x(metric), _k(metric), _kz(metric), _g(metric), _dD_dk(metric), _dD_dx(metric)
{
}
// Initialize via y[1] = (x_con[], k_cov[], other), note that y stars at 1
DispersionRelation::DispersionRelation(double y[], Metric& metric)
  : _x(metric), _k(metric), _kz(metric), _g(metric), _dD_dk(metric), _dD_dx(metric)
{
  reinitialize(y);
}
// Initialize via x and k
DispersionRelation::DispersionRelation(FourVector<double>& x, FourVector<double>& k, Metric& metric)
  : _x(metric), _k(metric), _kz(metric), _g(metric), _dD_dk(metric), _dD_dx(metric)
{
  reinitialize(x,k);
}
// Destructor
DispersionRelation::~DispersionRelation()
{
}


/*** REINITIALIZERS ***/
// Reinitialize via y
void DispersionRelation::reinitialize(double y[])
{
  _x.mkcon(y);
  _k.mkcov(y+4);
  reinitialize(_x,_k);
}
// Reinitialize (also does original initialization)
void DispersionRelation::reinitialize(FourVector<double>& x, 
				      FourVector<double>& k)
{
  _x = x;
  _k = k;
  defined=1;
  get_fcns();
}

/*** COMMON FUNCTION EVALUATION ***/
// Common Functions
void DispersionRelation::get_fcns() { }

/*** DISPERSION RELATION AND DERIVATIVES ***/
// DispersionRelation Relation
double DispersionRelation::D(int mode)
{
  if(defined%2){ // Check to see if _D is defined yet
    defined *= 2;
    _D = _k*_k;
  }
  return (_D);
}
double DispersionRelation::zeroing_D(int mode)
{
  return (D(mode));
}

// Derivative wrt k
FourVector<double>& DispersionRelation::dD_dk(int mode)
{
  if(defined%3){ // Check to see if _dD_dk is defined yet
    defined *= 3;
    _dD_dk = _k;
    _dD_dk *= 2.0;
  }
  return (_dD_dk);
}
// Derivative wrt x
FourVector<double>& DispersionRelation::dD_dx(int mode)
{
  if(defined%5){ // Check to see if _dD_dx is defined yet
    double dD[4]={0,0,0,0};
    defined *= 5;

    _g.Dginv(0);
    _k.cov(0);
    for(int i1=0;i1<_g.NDg;++i1){
      dD[_g.Dgk[i1]] += _k._cov[_g.Dgi[i1]]*_k._cov[_g.Dgj[i1]] * _g._Dginv[i1];
    }
    _dD_dx.mkcov(dD);
  }
  return (_dD_dx);
}


/*** DISPERSION RELATION ZEROING ***/
// Uses pointer for polymorphism
FourVector<double>& DispersionRelation::Zero_D(int mode)
{
  DispersionRelation D_local(_x,_k,_g);
  _zdisp_ptr = &D_local;
  return ( Zero_D_wptr(mode) );
}

#define NTRY 10000
FourVector<double>& DispersionRelation::Zero_D_wptr(int mode)
{
  int log=1;
  double kmag = 1.0;
  double k1 = 0.99, k2 =1.01;
  double tol = 1.0E-8;

  _kz = _k;
  _modez = mode;

  double zdf_kz = Zero_D_Func(1.0);

  if ( (zdf_kz*Zero_D_Func(k2)) < 0.0 )
    k1 = 1.0;
  else if ( (zdf_kz*Zero_D_Func(k1)) < 0.0 )
    k2 = 1.0;

 
  bool bad_root = false;
  //static FourVector<double> kz2(_g);
  FourVector<double> kz2(_g);
  double Dcheck;
  do {
    log=zbrac_AEB(k1, k2);
    if (log==0) {
      _kz.mkcov(0.0);
      return _kz;
    }

    kmag = zriddr_AEB(k1, k2, tol);

    // Check answer
    kz2.mkcov(_kz.cov(0),kmag*_kz.cov(1),kmag*_kz.cov(2),kmag*_kz.cov(3));
    _zdisp_ptr->reinitialize(_x,kz2);
    Dcheck = _zdisp_ptr->D(mode);
    
    // If answer has converged to non-zero
    if ( std::fabs(Dcheck) > 1.0 || vrt2_isnan(Dcheck) ) {
      // This was not a true root, probably from divergence at kmag, try to find others
      bad_root = true;
      // First try higher
      double k3 = 1.1*k2;

      if (!vrt2_isinf(k3))
	log = zbrac_AEB(k2,k3,-1);
      else
	log = 0;
      if (log==0) { // if no root was bracketed, then try lower
	k3 = 0.9*k1;
	if (k3==0.0)
	  log = zbrac_AEB(k3,k1,1);
	else
	  log = 0;
	if (log==0) {
	  _kz.mkcov(0.0);
	  return _kz;
	}
	else {
	  k2 = k1;
	  k1 = k3;
	}
      }
      else {
	k1 = k2;
	k2 = k3;
      }
    }
    else if (kmag==-999) { // if kmag = -999 throw exception stuff
      _kz.mkcov(0.0);
      return _kz;
    }
    else
      bad_root = false;
  }
  while (bad_root);

  _kz = kz2;

  return ( _kz );  
}
#undef NTRY
double DispersionRelation::Zero_D_Func(double kmag)
{
  // Set k.cov
  //static FourVector<double> k_local;
  FourVector<double> k_local;
  k_local.set_gptr(_g);
  k_local.mkcov(_kz.cov(0),kmag*_kz.cov(1),kmag*_kz.cov(2),kmag*_kz.cov(3));

  // Get new dispersion relation
  _zdisp_ptr->reinitialize(_x,k_local);

  return ( _zdisp_ptr->zeroing_D(_modez) );
}  

#define FACTOR 1.1
#define NTRY 1000
// If boundary>0 keep x2 fixed, if boundary<0 keep x1 fixed
//  if boundary=0 vary both (default)
int DispersionRelation::zbrac_AEB(double& x1, double& x2, int boundary)
{
  double f1,f2;
  
  if (x1 == x2){
    std::cerr << "Bad intial range in zbrac_AEB, x1=" << x1
	      << " x2=" << x2
	      << " boundary=" << boundary
	      << '\n';
    return -1;
  }
  f1=Zero_D_Func(x1);
  f2=Zero_D_Func(x2);
  int j;
  for (j=1;j<=NTRY;j++) {
    if (f1*f2 < 0.0)
      return j;
    if (boundary>=0)
      f1=Zero_D_Func(x1 /= FACTOR);
    if (boundary<=0)
      f2=Zero_D_Func(x2 *= FACTOR);
  }
  return 0;
}
#undef FACTOR
#undef NTRY 

#define MAXIT 10000
 
double DispersionRelation::rtflsp_AEB(double x1, double x2, double xacc)
{
  int j;
  double fl,fh,xl,xh,swap,dx,del,f,rtf;
 
  fl=Zero_D_Func(x1);
  fh=Zero_D_Func(x2);
 
  if (fl*fh > 0.0){
    std::cerr << "Root must be bracketed in rtflsp" << std::endl;
    return -999;
  }
  if (fl < 0.0) {
    xl=x1;
    xh=x2;
  }
  else {
    xl=x2;
    xh=x1;
    swap=fl;
    fl=fh;
    fh=swap;
  }
  dx=xh-xl;
  for (j=1;j<=MAXIT;j++) {
    rtf=xl+dx*fl/(fl-fh);
    f=Zero_D_Func(rtf);
 
    if (f < 0.0) {
      del=xl-rtf;
      xl=rtf;
      fl=f;
    }
    else {
      del=xh-rtf;
      xh=rtf;
      fh=f;
    }
    dx=xh-xl;
    if (std::fabs(del) < xacc || f == 0.0)
      return rtf;
  }
  std::cerr << "Maximum number of iterations exceeded in rtflsp" << std::endl;
  return -999;
}
#undef MAXIT

#define NRANSI
#define MAXIT 1000000
#define UNUSED (-1.11e30)
#define SIGN(a,b) ((b) >= 0.0 ? std::fabs(a) : -std::fabs(a)) 
 
double DispersionRelation::zriddr_AEB(double x1, double x2, double xacc)
{
  int j;
  double ans,fh,fl,fm,fnew,s,xh,xl,xm,xnew;

  fl=Zero_D_Func(x1);
  fh=Zero_D_Func(x2);

  if ((fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0)) {
    xl=x1;
    xh=x2;
    ans=UNUSED;
    for (j=1;j<=MAXIT;j++) {
      xm=0.5*(xl+xh);
      fm=Zero_D_Func(xm);
      s=std::sqrt(fm*fm-fl*fh);
      if (s == 0.0)
	return ans;
      xnew=xm+(xm-xl)*((fl >= fh ? 1.0 : -1.0)*fm/s);
      if (std::fabs(xnew-ans) <= xacc)
	return ans;
      ans=xnew;
      fnew=Zero_D_Func(ans);
      if (fnew == 0.0)
	return ans;
      if (SIGN(fm,fnew) != fm) {
	xl=xm;
	fl=fm;
	xh=ans;
	fh=fnew;
      }
      else if (SIGN(fl,fnew) != fl) {
	xh=ans;
	fh=fnew;
      }
      else if (SIGN(fh,fnew) != fh) {
	xl=ans;
	fl=fnew;
      }
      else
	std::cerr << "never get here in zridder." << std::endl;
      if (std::fabs(xh-xl) <= xacc)
	return ans;
    }
    std::cerr << "Maximum number of iterations exceeded in zriddr" << std::endl;
    return -999;
  }
  else {
    if (fl == 0.0)
      return x1;
    if (fh == 0.0)
      return x2;
    std::cerr << "Root not bracketed in zridder\n";
    return -999;
  }
  std::cerr << "never get here 2\n";
  return 0.0;
}
#undef SIGN
#undef MAXIT
#undef UNUSED
#undef NRANSI
};
