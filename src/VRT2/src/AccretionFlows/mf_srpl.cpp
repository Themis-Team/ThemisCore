#include "mf_srpl.h"

namespace VRT2 {

MF_SRPL::MF_SRPL(Metric& g, double B0, double power)
  : MagneticField(g), _B0(B0), _power(power)
{
}

FourVector<double>& MF_SRPL::get_field_fourvector(double t, double r, double theta, double phi)
{
  double sgn = (std::cos(theta)<0 ? -1 : 1);
  _b.mkcon(0.0,sgn*_B0*std::pow(r,_power),0,0);

  return _b;
}

};
