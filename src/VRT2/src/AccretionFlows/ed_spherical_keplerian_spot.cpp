#include "ed_spherical_keplerian_spot.h"

namespace VRT2 {
ED_SphericalKeplerianSpot::ED_SphericalKeplerianSpot(Metric& g, double density_scale, double rspot, double rorbit, double height, double phi0)
  : _g(g), _density_scale(density_scale), _rspot(rspot), _rhoorbit(rorbit), _zorbit(height), _phi0(phi0),
     _uspot(_g), _xspot_center(_g)
{
  _Omega = 1.0/( std::pow(_rhoorbit/_g.mass(),1.5) + _g.ang_mom() );

  _rorbit = std::sqrt(_rhoorbit*_rhoorbit + _zorbit*_zorbit);
  _thetaorbit = atan2(_rhoorbit,_zorbit);
}
};
