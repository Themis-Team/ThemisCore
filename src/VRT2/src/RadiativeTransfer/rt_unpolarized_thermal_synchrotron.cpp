#include "rt_unpolarized_thermal_synchrotron.h"


namespace VRT2 {
RT_UnpolarizedThermalSynchrotron::RT_UnpolarizedThermalSynchrotron(Metric& g,
					     ElectronDensity& ne, Temperature& Te,
					     AccretionFlowVelocity& u, MagneticField& B)
  : RadiativeTransfer(g), _ne(ne), _Te(Te), _u(u), _B(B)
{
  generate_K2_lookup_table();
  set_constants();
}
RT_UnpolarizedThermalSynchrotron::RT_UnpolarizedThermalSynchrotron(const double y[], Metric& g,
					     ElectronDensity& ne, Temperature& Te,
					     AccretionFlowVelocity& u, MagneticField& B)
  : RadiativeTransfer(y,g), _ne(ne), _Te(Te), _u(u), _B(B)
{
  generate_K2_lookup_table();
  set_constants();
}
RT_UnpolarizedThermalSynchrotron::RT_UnpolarizedThermalSynchrotron(FourVector<double>& x, FourVector<double>& k, Metric& g,
					     ElectronDensity& ne, Temperature& Te,
					     AccretionFlowVelocity& u, MagneticField& B)
  : RadiativeTransfer(x,k,g), _ne(ne), _Te(Te), _u(u), _B(B)
{
  generate_K2_lookup_table();
  set_constants();
}

void RT_UnpolarizedThermalSynchrotron::set_frequency_scale(double omega0)
{
  RadiativeTransfer::set_frequency_scale(omega0);
  set_constants();
}

void RT_UnpolarizedThermalSynchrotron::set_length_scale(double L)
{
  RadiativeTransfer::set_length_scale(L);
  set_constants();
}

void RT_UnpolarizedThermalSynchrotron::reinitialize(const double y[])
{
  _x.mkcon(y);
  _k.mkcov(y+4);
  set_common_funcs();
}

void RT_UnpolarizedThermalSynchrotron::reinitialize(FourVector<double>& x, FourVector<double>& k)
{
  _x = x;
  _k = k;
  set_common_funcs();
}

void RT_UnpolarizedThermalSynchrotron::set_constants()
{
  // Emission constant (in cgs units)
  _Cjnu = 2.0 * VRT2_Constants::e*VRT2_Constants::e / ( std::sqrt(3.0)*VRT2_Constants::c);

  // Absorption constant (in cgs units)
  _Calphanu = 2.0*VRT2_Constants::pi*VRT2_Constants::pi / VRT2_Constants::me;

  // Cyclotron frequency coefficients omega_B = _ComegaB * B * sin(theta)
  _ComegaB = VRT2_Constants::e / (VRT2_Constants::me * VRT2_Constants::c);

  // Electron temperature coefficient theta_e = _Cthetae * Te
  _Cthetae = VRT2_Constants::k / (VRT2_Constants::me * VRT2_Constants::c*VRT2_Constants::c);

  // Put length into length scale as this is what dl is in
  _Cjnu *= _length_scale;
  // Note that since _Calphanu takes j_nu to alpha_nu, it already is taken care of by _Cjnu
  //_Calphanu *= _length_scale;
}

void RT_UnpolarizedThermalSynchrotron::set_common_funcs()
{
  FourVector<double> b = _B(_x);
  double omegaB = _ComegaB * std::sqrt( (b*b) ); // Get omega_B
  double othetae = (_Cthetae * _Te(_x)); // Get electron temperature

  if (omegaB>0.0 && othetae>0.0) {
    othetae = 1.0/othetae;//std::min(1.0/othetae,5.0e15); // Defines a minimum non-zero temperature (~1 microK)
    FourVector<double> u = _u(_x);
    double omega = -_omega_scale * (_k*u); // Get proper omega

    // Get emission and absorption coefficients from
    //  Mahadevan, Narayan & Yi, 1996, ApJ, 465, 327
    //  Yuan, Quataert & Narayan, 2003, ApJ, 598, 301
    double xM = 2.0*omega*othetae*othetae/(3.0*omegaB);
    double Mmexp = 4.0505*std::pow(xM,-1.0/6.0)
      * ( 1.0 + 0.40*std::pow(xM,-0.25) + 0.5316*std::pow(xM,-0.5) );
    double Mexpterms = std::exp( -1.8899*std::pow(xM,1.0/3.0) - lnK2(othetae) );
    _jnu = _Cjnu * _ne(_x) * Mmexp * Mexpterms / (omega*omega);
    _alphanu = _Calphanu * omega * _jnu * othetae;
  }
  else {
    _jnu = 0.0;
    _alphanu = 0.0;
  }
}

double RT_UnpolarizedThermalSynchrotron::isotropic_absorptivity(const double dydx[])
{
  return _alphanu * dl_dlambda(dydx);
}

std::valarray<double>& RT_UnpolarizedThermalSynchrotron::IQUV_abs(const double iquv[], const double dydx[])
{
  if (_alphanu != 0.0) {
    double abs = _alphanu * dl_dlambda(dydx);
    _iquv_abs[0] = abs * iquv[0];
    _iquv_abs[1] = abs * iquv[1];
    _iquv_abs[2] = abs * iquv[2];
    _iquv_abs[3] = abs * iquv[3];
  }
  else
    _iquv_abs = 0.0;
  return _iquv_abs;
}

std::valarray<double>& RT_UnpolarizedThermalSynchrotron::IQUV_ems(const double dydx[])
{
  if (_jnu != 0.0 ) {
    _iquv_ems[0] = _jnu * dl_dlambda(dydx);
    _iquv_ems[1] = 0.0;
    _iquv_ems[2] = 0.0;
    _iquv_ems[3] = 0.0;
  }
  else
    _iquv_ems = 0.0;
  return _iquv_ems;
}

double RT_UnpolarizedThermalSynchrotron::dl_dlambda(const double dydx[])
{
  // Note that dx_dlam^2 = 0 b.c. this is a null geodesic!
  FourVector<double> dx_dlam(_g);
  dx_dlam.mkcon(dydx);

  return std::fabs(dx_dlam*_u(_x));
}

// These are designed to speed up the bessel function K2 evaluations substantially.
// The arrays _x_K2 and _K2_K2 contains values of ln(x) and ln(K2), respectively,
// so that interpolation is performed in log-log plane to capture asymptotic x->0
// behavior.
void RT_UnpolarizedThermalSynchrotron::generate_K2_lookup_table()
{
  // HARD WIRED K2 ARGUMENTS LIMITS (INCLUSIVE)
  _x_K2_min = -2*std::log(10.0);
  _x_K2_max = 2.86*std::log(10.0); // close enough to minimum double representable on 32-bit

  // Choose number of elements and fill lookup tables
  size_t N = 101;
  _x_K2_stp = (_x_K2_max-_x_K2_min)/double(N-1);
  _x_K2.resize(N);
  _K2_K2.resize(N);
  for (size_t i=0; i<N; ++i) {
    _x_K2[i] = _x_K2_min + _x_K2_stp*i;
    _K2_K2[i] = std::log(bessk(2,std::exp(_x_K2[i])));
  }

  /*
  // DEBUG PICTURE CHECK
  for (size_t i=0; i<N; ++i)
    std::cout << std::setw(15) << std::exp(_x_K2[i])
	      << std::setw(15) << std::exp(_K2_K2[i])
	      << std::endl;
  std::cout << "\n\n";
  for (int i=0; i<11000; ++i)
    std::cout << std::setw(15) << i/10.0
	      << std::setw(15) << lnK2(i/10.0)
	      << std::endl;
  */
}
// Interpolation access function for K2 tables (return log of K2)
double RT_UnpolarizedThermalSynchrotron::lnK2(double x)
{
  x = std::log(x);
  if (x<_x_K2_min)
    return _K2_K2[0];
  else if (x>_x_K2_max)
    return _K2_K2[_K2_K2.size()-1];
  else {
    size_t lower_index = size_t(floor((x-_x_K2_min)/_x_K2_stp));
    double dx = (x - _x_K2[lower_index])/_x_K2_stp;
    return dx*_K2_K2[lower_index+1] + (1.0-dx)*_K2_K2[lower_index];
  }
}
double RT_UnpolarizedThermalSynchrotron::bessi0(double x)
{
  double ax,ans;
  double y;

  if ((ax=fabs(x)) < 3.75) {
    y=x/3.75;
    y*=y;
    ans=1.0+y*(3.5156229+y*(3.0899424
			    +y*(1.2067492
				+y*(0.2659732
				    +y*(0.360768e-1
					+y*0.45813e-2)))));
  } else {
    y=3.75/ax;
    ans=(exp(ax)/sqrt(ax))*(0.39894228
			    +y*(0.1328592e-1
				+y*(0.225319e-2
				    +y*(-0.157565e-2
					+y*(0.916281e-2
					    +y*(-0.2057706e-1
						+y*(0.2635537e-1
						    +y*(-0.1647633e-1
							+y*0.392377e-2))))))));
  }
  return ans;
}
double RT_UnpolarizedThermalSynchrotron::bessk0(double x)
{
  double y,ans;

  if (x <= 2.0) {
    y=x*x/4.0;
    ans=(-log(x/2.0)*bessi0(x))+(-0.57721566
				 +y*(0.42278420
				     +y*(0.23069756
					 +y*(0.3488590e-1
					     +y*(0.262698e-2
						 +y*(0.10750e-3
						     +y*0.74e-5))))));
  } else {
    y=2.0/x;
    ans=(exp(-x)/sqrt(x))*(1.25331414
			   +y*(-0.7832358e-1
			       +y*(0.2189568e-1
				   +y*(-0.1062446e-1
				       +y*(0.587872e-2
					   +y*(-0.251540e-2
					       +y*0.53208e-3))))));
  }
  return ans;
}
double RT_UnpolarizedThermalSynchrotron::bessi1(double x)
{
  double ax,ans;
  double y;

  if ((ax=fabs(x)) < 3.75) {
    y=x/3.75;
    y*=y;
    ans=ax*(0.5+y*(0.87890594
		   +y*(0.51498869
		       +y*(0.15084934
			   +y*(0.2658733e-1
			       +y*(0.301532e-2
				   +y*0.32411e-3))))));
  } else {
    y=3.75/ax;
    ans=0.2282967e-1+y*(-0.2895312e-1
			+y*(0.1787654e-1
			    -y*0.420059e-2));
    ans=0.39894228+y*(-0.3988024e-1
		      +y*(-0.362018e-2
			  +y*(0.163801e-2
			      +y*(-0.1031555e-1
				  +y*ans))));
    ans *= (exp(ax)/sqrt(ax));
  }
  return x < 0.0 ? -ans : ans;
}
double RT_UnpolarizedThermalSynchrotron::bessk1(double x)
{
  double y,ans;

  if (x <= 2.0) {
    y=x*x/4.0;
    ans=(log(x/2.0)*bessi1(x))+(1.0/x)*(1.0
					+y*(0.15443144
					    +y*(-0.67278579
						+y*(-0.18156897
						    +y*(-0.1919402e-1
							+y*(-0.110404e-2
							    +y*(-0.4686e-4)))))));
  } else {
    y=2.0/x;
    ans=(exp(-x)/sqrt(x))*(1.25331414
			   +y*(0.23498619
			       +y*(-0.3655620e-1
				   +y*(0.1504268e-1
				       +y*(-0.780353e-2
					   +y*(0.325614e-2
					       +y*(-0.68245e-3)))))));
  }
  return ans;
}
double RT_UnpolarizedThermalSynchrotron::bessk(int n, double x)
{
  int j;
  double bk,bkm,bkp,tox;

  tox=2.0/x;
  bkm=bessk0(x);
  bk=bessk1(x);
  for (j=1;j<n;j++) {
    bkp=bkm+j*tox*bk;
    bkm=bk;
    bk=bkp;
  }
  return bk;
}
};
