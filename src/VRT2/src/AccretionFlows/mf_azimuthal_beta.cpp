#include "mf_azimuthal_beta.h"


namespace VRT2 {
MF_AzimuthalBeta::MF_AzimuthalBeta(Metric& g, AccretionFlowVelocity& u, ElectronDensity& ne, double beta)
  : MagneticField(g), _u(u), _ne(ne), _beta(beta)
{
}

FourVector<double>& MF_AzimuthalBeta::get_field_fourvector(double t, double r, double theta, double phi)
{
  // Get proper magnitude (=b^2)
  double P = _ne(t,r,theta,phi) * (_g.mass()/r) * 1.5e-3 / 6.0; // m_p c^2 = 1.5e-3 erg, 6 from virial thm
  double Bmag = std::sqrt( 8.0*VRT2_Constants::pi * P / _beta );
  double tmpval;

  //Bmag = std::sqrt( (_ne(t,r,theta,phi) * _g.mass()/r) / _beta );

  // Get direction orthogonal to flow velocity along z direction
  FourVector<double> utmp = _u(t,r,theta,phi);
  _b.mkcon(0.0,std::cos(theta),-std::sin(theta)/r,0.0);
  _b = _b + (utmp*_b)*utmp;
  tmpval = _b*_b;
  _b *= Bmag/std::sqrt((tmpval));

  return _b;
}
};
