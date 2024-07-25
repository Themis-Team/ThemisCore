#include "rt_unpolarized_power_law_synchrotron.h"


namespace VRT2 {
RT_UnpolarizedPowerLawSynchrotron::RT_UnpolarizedPowerLawSynchrotron(Metric& g,
					       ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
					       double spectral_index, double gamma_min)
  : RadiativeTransfer(g), _ne(ne), _u(u), _B(B),
     _spectral_index(spectral_index), _gamma_min(gamma_min)
{
  set_constants();
}
RT_UnpolarizedPowerLawSynchrotron::RT_UnpolarizedPowerLawSynchrotron(const double y[], Metric& g,
					       ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
					       double spectral_index, double gamma_min)
  : RadiativeTransfer(y,g), _ne(ne), _u(u), _B(B),
     _spectral_index(spectral_index), _gamma_min(gamma_min)
{
  set_constants();
}
RT_UnpolarizedPowerLawSynchrotron::RT_UnpolarizedPowerLawSynchrotron(FourVector<double>& x, FourVector<double>& k, Metric& g,
					       ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
					       double spectral_index, double gamma_min)
  : RadiativeTransfer(x,k,g), _ne(ne), _u(u), _B(B),
     _spectral_index(spectral_index), _gamma_min(gamma_min)
{
  set_constants();
}

void RT_UnpolarizedPowerLawSynchrotron::set_frequency_scale(double omega0)
{
  RadiativeTransfer::set_frequency_scale(omega0);
  set_constants();
}

void RT_UnpolarizedPowerLawSynchrotron::set_length_scale(double L)
{
  RadiativeTransfer::set_length_scale(L);
  set_constants();
}

void RT_UnpolarizedPowerLawSynchrotron::reinitialize(const double y[])
{
  _x.mkcon(y);
  _k.mkcov(y+4);
  set_common_funcs();
}

void RT_UnpolarizedPowerLawSynchrotron::reinitialize(FourVector<double>& x, FourVector<double>& k)
{
  _x = x;
  _k = k;
  set_common_funcs();
}

void RT_UnpolarizedPowerLawSynchrotron::set_constants()
{
  // Emission constant (in cgs units)
  _Cjnu = VRT2_Constants::me * VRT2_Constants::c * VRT2_Constants::re
    * std::pow( 3.0, _spectral_index + 0.5 ) * std::exp(gammln(0.5*_spectral_index + 11.0/6.0) + gammln(0.5*_spectral_index + 1.0/6.0))
    / (8.0*VRT2_Constants::pi*(_spectral_index + 1.0));

  // Absorption constant (in cgs units)
  _Calphanu = 2.0 * VRT2_Constants::pi * VRT2_Constants::re * VRT2_Constants::c
    * std::pow( 3.0, _spectral_index + 1.0 ) * std::exp(gammln(0.5*_spectral_index + 25.0/12.0) * gammln(0.5*_spectral_index + 5.0/12.0))
    / 4.0;

  // Stokes Q ratios to Stokes I quantities
  _epsilon_Q = 0.0; //(_spectral_index+1.0)/(_spectral_index+5.0/3.0);
  _zeta_Q = 0.0; //(_spectral_index+1.5)/(_spectral_index+13.0/6.0);

  // Cyclotron frequency coefficients omega_B = _ComegaB * B * sin(theta)
  _ComegaB = VRT2_Constants::e / (VRT2_Constants::me * VRT2_Constants::c);
  // Density constant n_gamma = n0 gamma^-s, n0 = _Cn n_tot
  _Cn = 2.0*_spectral_index*std::pow(_gamma_min,2*_spectral_index+1);

  // Put length into length scale as this is what dl is in
  _Cjnu *= _length_scale;
  _Calphanu *= _length_scale;
}

void RT_UnpolarizedPowerLawSynchrotron::set_common_funcs()
{
  _n0 = _Cn * _ne(_x);

  FourVector<double> u = _u(_x);
  _omega = - (_k*u); // Get scaled omega


  FourVector<double> b = _B(_x);
  double bmag = std::sqrt( (b*b) );
  //_sn_alpha = (bmag>0 ? (b*_k)/(bmag * _omega) : 1.0); // using k^2=0
  //_sn_alpha = std::sqrt( std::max(0.0,1.0 - _sn_alpha*_sn_alpha) ); // deals with catastrophic subtraction

  // HACK TO CHECK ANGULAR DEPENDENCE AT LOW INCLINATIONS (FACE-ON)
  _sn_alpha = std::sqrt(2.0/3.0);

  _omegaB = _ComegaB * bmag;

  _omega *= _omega_scale; // Get real omega instead of scaled omega

  // Get rotation coeffs to align Stokes bases
  get_Stokes_alignment_angle(u,b,_cs,_sn);
}

void RT_UnpolarizedPowerLawSynchrotron::get_Stokes_alignment_angle(FourVector<double>& u, FourVector<double>& b, double& cs, double& sn)
{
  FourVector<double> uZAMO(_g), z(_g);
  uZAMO.mkcov(1.0,0.0,0.0,0.0);
  z.mkcon(0.0,std::cos(_x.con(2)),-std::sin(_x.con(2))/_x.con(1),0.0);

  FourVector<double> eperp = cross_product(uZAMO,_k,z);
  eperp *= 1.0/std::sqrt( (eperp*eperp) );
  FourVector<double> ebperp = cross_product(u,_k,b);
  ebperp *= 1.0/std::sqrt( (ebperp*ebperp) );
  FourVector<double> ebpara = cross_product(u,_k,ebperp);
  ebpara *= 1.0/std::sqrt( (ebpara*ebpara) );

  double cstmp = (eperp*ebperp);
  double sntmp = (eperp*ebpara); // Check in comparision to choice of cross product

  // Use recursion relations to get cos(2*phi), sin(2*phi)
  cs = cstmp*cstmp - sntmp*sntmp;
  sn = 2.0*cstmp*sntmp;
}

double RT_UnpolarizedPowerLawSynchrotron::isotropic_absorptivity(const double dydx[])
{
  if (_omegaB > 0.0)
    return _Calphanu * (_n0/_omega) * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index+2.5-1) * dl_dlambda(dydx);
  else
    return 0.0;
}

std::valarray<double>& RT_UnpolarizedPowerLawSynchrotron::IQUV_abs(const double iquv[], const double dydx[])
{
  if (_omegaB > 0.0) {
    double consts = _Calphanu * (_n0/_omega) * dl_dlambda(dydx);
    double abs = consts * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index+2.5-1);
    double qabs = _cs * _zeta_Q * abs;
    double uabs = _sn * _zeta_Q * abs;

    _iquv_abs[0] = abs*iquv[0] + qabs*iquv[1] + uabs*iquv[2];
    _iquv_abs[1] = qabs*iquv[0] + abs*iquv[1];
    _iquv_abs[2] = uabs*iquv[0] + abs*iquv[2];
    _iquv_abs[3] = abs*iquv[3];
  }
  else
    _iquv_abs = 0.0;

  return _iquv_abs;
}

std::valarray<double>& RT_UnpolarizedPowerLawSynchrotron::IQUV_ems(const double dydx[])
{
  if (_omegaB > 0.0) {
    double consts = _Cjnu * _omegaB * _n0 * std::pow(_omega,-3.0) * dl_dlambda(dydx);
    double ems = consts * _sn_alpha * std::pow(_omegaB*_sn_alpha/_omega,_spectral_index);
    _iquv_ems[0] = ems;
    _iquv_ems[1] = _cs * _epsilon_Q * ems;
    _iquv_ems[2] = _sn * _epsilon_Q * ems;
  }
  else
    _iquv_ems = 0.0;

  return _iquv_ems;
}

double RT_UnpolarizedPowerLawSynchrotron::dl_dlambda(const double dydx[])
{
  // Note that dx_dlam^2 = 0 b.c. this is a null geodesic!
  FourVector<double> dx_dlam(_g);
  dx_dlam.mkcon(dydx);

  return std::fabs(dx_dlam*_u(_x));
}

double RT_UnpolarizedPowerLawSynchrotron::gammln(double xx)
{
  double x,y,tmp,ser;
  static double cof[6]={76.18009172947146,-86.50532032941677,
			24.01409824083091,-1.231739572450155,
			0.1208650973866179e-2,-0.5395239384953e-5};
  int j;

  y=x=xx;
  tmp=x+5.5;
  tmp -= (x+0.5)*log(tmp);
  ser=1.000000000190015;
  for (j=0;j<=5;j++)
    ser += cof[j]/++y;
  return -tmp+log(2.5066282746310005*ser/x);
}
};
