#include "mf_split_monopole.h"

namespace VRT2 {

MF_SplitMonopole::MF_SplitMonopole(Metric& g, AccretionFlowVelocity& u, double B0)
  : MagneticField(g), _u(u), _B0(B0)
{
}

FourVector<double>& MF_SplitMonopole::get_field_fourvector(double t, double r, double theta, double phi)
{
  double sgn = (std::cos(theta)<0 ? -1 : 1);
  FourVector<double> utmp = _u(t,r,theta,phi);
  _b.mkcon(0.0,sgn*_B0*utmp.cov(0)/(r*r),0,0);

  return _b;
}

};
