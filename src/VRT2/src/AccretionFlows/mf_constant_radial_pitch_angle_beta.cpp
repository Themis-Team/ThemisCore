#include "mf_constant_radial_pitch_angle_beta.h"

namespace VRT2 {

MF_ConstantRadialPitchAngleBeta::MF_ConstantRadialPitchAngleBeta(Metric& g, AccretionFlowVelocity& u, ElectronDensity& ne, double angle, double beta)
  : MagneticField(g), _u(u), _ne(ne), _sa(std::sin(angle)), _ca(std::cos(angle)), _beta(beta)
{
}

FourVector<double>& MF_ConstantRadialPitchAngleBeta::get_field_fourvector(double t, double r, double theta, double phi)
{
  // Get proper magnitude (=b^2)
  double P = _ne(t,r,theta,phi) * (_g.mass()/r) * 1.5e-3 / 6.0; // m_p c^2 = 1.5e-3 erg, 6 from virial thm
  double Bmag = std::sqrt( 8.0*VRT2_Constants::pi * P / _beta );
  double tmp;

  //Bmag = std::sqrt( (_ne(t,r,theta,phi) * _g.mass()/r) / _beta );

  // Get direction orthogonal to flow velocity along phi direction
  FourVector<double> utmp = _u(t,r,theta,phi);
  //_b.mkcon(0.0,std::cos(theta)*_sa,-std::sin(theta)*_sa/r,_ca);
  _b.mkcon(0.0,std::sin(theta)*_sa,std::cos(theta)*_sa/r,_ca);
  _b = _b + (utmp*_b)*utmp;
  tmp = _b*_b;
  _b *= Bmag/sqrt((tmp));

  return _b;
}

};
