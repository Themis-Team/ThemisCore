#include "SgrA_disk_model_parameters.h"

//FIT DISK MODEL PARAMS

namespace VRT2 {
SgrA_FitDiskModelParameters::SgrA_FitDiskModelParameters(double a, double THETA)
  : _a(a), _THETA(THETA)
{
  nea_[0] = 21.0204927588119;
  nea_[1] = -47.8481793371556;
  nea_[2] = 36.3963450741395;
  nea_[3] = -11.1259374419478;

  neb_[0] = -15.7183355056535;
  neb_[1] = 32.8488871594679;
  neb_[2] = -20.3626289178381;
  neb_[3] = 3.05351770893372;

  nec_[0] = 0.92360429795355;
  nec_[1] = 0.82653378805878;
  nec_[2] = 0.173119523449205;
  nec_[3] = 16.9629822676377;


  Tea_[0] = 22.4707236735606;
  Tea_[1] = -38.0391352529862;
  Tea_[2] = 17.2468598541929;
  Tea_[3] = -0.938686717844435;

  Teb_[0] = -7.44252883724679;
  Teb_[1] = 12.2116907739415;
  Teb_[2] = -5.85318316585477;
  Teb_[3] = 0.836155129605212;

  Tec_[0] = -0.913184556153671;
  Tec_[1] = 0.661475120794941;
  Tec_[2] = 0.0141526568232037;
  Tec_[3] = 25.2683304948649;


  nntha_[0] = -19.49086303635;
  nntha_[1] = 44.6555910444129;
  nntha_[2] = -34.2519679440164;
  nntha_[3] = 10.2733040784462;

  nnthb_[0] = 13.8154084464426;
  nnthb_[1] = -29.5439996630622;
  nnthb_[2] = 18.7829471840758;
  nnthb_[3] = -3.09602136789871;

  nnthc_[0] = 0.398654881782504;
  nnthc_[1] = -2.38420032478941;
  nnthc_[2] = -0.153446914563209;
  nnthc_[3] = 11.5948693372493;
}
void SgrA_FitDiskModelParameters::reset(double a, double THETA)
{
  _a = a;
  _THETA = THETA;
}

double SgrA_FitDiskModelParameters::ne_norm() const
{
  double a[4];
  for (int i=0; i<4; ++i)
    a[i] = nec_[i] + _a*( neb_[i] + _a*nea_[i] );
  double mu=std::cos(M_PI*_THETA/180.0);
  double val=a[0];
  for (int i=1; i<4; ++i)
    val = val*mu + a[i];

  return std::exp(val);
}
double SgrA_FitDiskModelParameters::Te_norm() const
{
  double a[4];
  for (int i=0; i<4; ++i)
    a[i] = Tec_[i] + _a*( Teb_[i] + _a*Tea_[i] );
  double mu=std::cos(M_PI*_THETA/180.0);
  double val=a[0];
  for (int i=1; i<4; ++i)
    val = val*mu + a[i];

  return std::exp(val);
}
double SgrA_FitDiskModelParameters::nnth_norm() const
{
  double a[4];
  for (int i=0; i<4; ++i)
    a[i] = nnthc_[i] + _a*( nnthb_[i] + _a*nntha_[i] );
  double mu=std::cos(M_PI*_THETA/180.0);
  double val=a[0];
  for (int i=1; i<4; ++i)
    val = val*mu + a[i];

  return std::exp(val);
}

//POINT DISK MODEL PARAMS

SgrA_PolintDiskModelParameters::SgrA_PolintDiskModelParameters(std::string fname, int aorder, int thetaorder, double a, double THETA)
  : _a(a), _THETA(THETA), _aorder(aorder), _thetaorder(thetaorder)
{
  // Read data from file into tables
  std::ifstream in(fname.c_str());
  if (!in.is_open()) {
    std::cerr << "Couldn't open " << fname << std::endl;
    std::abort();
  }

  in.ignore(4096,'\n');
  double tmp;
  do {
    in >> tmp;
    a_.push_back(tmp);
    in >> tmp;
    theta_.push_back(tmp);
    in >> tmp;
    ne_.push_back(std::log(tmp));
    in >> tmp;
    Te_.push_back(std::log(tmp));
    in >> tmp;
    nnth_.push_back(std::log(tmp));
    in.ignore(4096,'\n');
  } while (!in.eof());
  a_.pop_back();
  theta_.pop_back();
  ne_.pop_back();
  Te_.pop_back();
  nnth_.pop_back();

  // Get table limits (assumes that a & theta are listed in ascending order)
  amin_ = amax_ = a_[0];
  thetamin_ = thetamax_ = theta_[0];
  Na_ = 1;
  Ntheta_ = 1;
  for (size_t i=1; i<a_.size(); ++i) {
    if (a_[i]>amax_) {
      ++Na_;
      amax_ = a_[i];
    }
    if (theta_[i]>thetamax_) {
      ++Ntheta_;
      thetamax_ = theta_[i];
    }
  }
}
void SgrA_PolintDiskModelParameters::reset(double a, double THETA)
{
  _a = a;
  _THETA = THETA;
}

void SgrA_PolintDiskModelParameters::set_orders(int aorder, int thetaorder)
{
  _aorder = aorder;
  _thetaorder = thetaorder;
}

double SgrA_PolintDiskModelParameters::ne_norm() const
{
  return (  std::exp( interpolate2D(ne_,_aorder,_thetaorder,_a,_THETA) )  );
}
double SgrA_PolintDiskModelParameters::Te_norm() const
{
  return (  std::exp( interpolate2D(Te_,_aorder,_thetaorder,_a,_THETA) )  );
}
double SgrA_PolintDiskModelParameters::nnth_norm() const
{
  return (  std::exp( interpolate2D(nnth_,_aorder,_thetaorder,_a,_THETA) )  );
}


double SgrA_PolintDiskModelParameters::interpolate2D(const std::vector<double>& y, int na, int nth, double a, double theta) const
{
  // 1st get interpolation limits
  //  (A) Get stencil size to fill
  if (nth>Ntheta_) // Can only use the maximum number of points
    nth = Ntheta_;
  if (na>Na_)
    na = Na_;
  //  (B) Assuming that a_ & theta_ are in ascending order, get index of first value less than one of interest
  int ith=0, ia=0;
  for (ith=0; ith<Ntheta_ && theta>theta_[ith]; ++ith) {};
  for (ia=0; ia<Na_ && a>a_[ia*Ntheta_]; ++ia) {};
  //  (C) Try to center stencil on this value
  int ith0 = std::min(std::max(ith-nth/2,0),Ntheta_-nth);
  int ia0 = std::min(std::max(ia-na/2,0),Na_-na);

  // Allocate work arrays
  double *xth = new double[nth+1];
  double *yth = new double[nth+1];
  double *xa = new double[na+1];
  double *ya = new double[na+1];

  // Do interpolations
  int index_offset;
  double dy;
  for (ia=0; ia<na; ++ia) {
    index_offset = (ia0+ia)*Ntheta_;
    // (A) For each a in stencil interpolate on theta
    for (ith=0; ith<nth; ++ith) {
      xth[ith+1] = theta_[index_offset + ith0 + ith];
      yth[ith+1] = y[index_offset + ith0 + ith];
    }
    polint(xth,yth,nth,theta,ya[ia+1],dy);
    xa[ia+1] = a_[index_offset];
  }
  double val;
  polint(xa,ya,na,a,val,dy);

  return val;
}

// Numerical Recipes: Interpolates x into the arrays xa,ya to get y with
//   error estimate dy after doing nth-order polynomial interpolation.
// NOTE THAT THIS USES UNIT-OFFSET ARRAYS (YUCK!!!)
void SgrA_PolintDiskModelParameters::polint(double xa[], double ya[], int n, double x, double &y, double &dy) const
{
  int i,m,ns=1;
  double den,dif,dift,ho,hp,w;
  double *c=new double[n+1];
  double *d=new double[n+1];

  dif=fabs(x-xa[1]);
  for (i=1;i<=n;i++) {
    if ( (dift=fabs(x-xa[i])) < dif) {
      ns=i;
      dif=dift;
    }
    c[i]=ya[i];
    d[i]=ya[i];
  }
  y=ya[ns--];
  for (m=1;m<n;m++) {
    for (i=1;i<=n-m;i++) {
      ho=xa[i]-x;
      hp=xa[i+m]-x;
      w=c[i+1]-d[i];
      if ( (den=ho-hp) == 0.0)
	std::cerr << "Error in routine polint\n";
      den=w/den;
      d[i]=hp*den;
      c[i]=ho*den;
    }
    y += (dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
  }
  delete[] c;
  delete[] d;
}




};
