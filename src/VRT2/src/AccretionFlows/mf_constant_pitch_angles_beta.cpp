#include "mf_constant_pitch_angles_beta.h"

namespace VRT2 {
MF_ConstantPitchAnglesBeta::MF_ConstantPitchAnglesBeta(Metric& g, AccretionFlowVelocity& u, ElectronDensity& ne, double angleBphi, double angleZ, double beta)
  : MagneticField(g), _u(u), _ne(ne), _sa1(std::sin(angleBphi)), _ca1(std::cos(angleBphi)), _sa2(std::sin(angleZ)), _ca2(std::cos(angleZ)), _beta(beta)
{
}

FourVector<double>& MF_ConstantPitchAnglesBeta::get_field_fourvector(double t, double r, double theta, double phi)
{
  // Get proper magnitude (=b^2)
  double P = _ne(t,r,theta,phi) * (_g.mass()/r) * 1.5e-3 / 6.0; // m_p c^2 = 1.5e-3 erg, 6 from virial thm
  double Bmag = std::sqrt( 8.0*VRT2_Constants::pi * P / _beta );
  double tmpval;

  //Bmag = std::sqrt( (_ne(t,r,theta,phi) * _g.mass()/r) / _beta );

  // Get direction orthogonal to flow velocity along phi direction
  double st = std::sin(theta);
  double ct = std::cos(theta);

  FourVector<double> utmp = _u(t,r,theta,phi);

  // zcyl = ct rhat - st/r thhat
  // rcyl = st rhat + ct/r thhat

  // BP = ca2*zcyl + sa2*rcl (clockwise for phi=0 on right side)
  //    = ca2*(ct rhat - st/r thhat) + sa2*(st rhat + ct/r thhat)
  //    = (ca2 ct + sa2 st )*rhat + ( -ca2 st + sa2 ct )/r thhat

  _b.mkcon(0.0,_sa1*(_ca2*ct+_sa2*st), _sa1*(_sa2*ct-_ca2*st)/r,_ca1);
  _b = _b + (utmp*_b)*utmp;
  tmpval = _b*_b;
  _b *= Bmag/std::sqrt(tmpval);

  return _b;
}
};
