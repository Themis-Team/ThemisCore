#include "ed_spherical_keplerian_stochastic_spot.h"


namespace VRT2 {
ED_SphericalKeplerianStochasticSpot::ED_SphericalKeplerianStochasticSpot(Metric& g, double dm, double sigld, double Dt, double t0, double rspot, double rorbit, double height, double phi0)
: _g(g), _ldm(std::log10(dm)), _sigld(sigld), _Dt(Dt), _t0(t0), _rspot(rspot), _rhoorbit(rorbit), _zorbit(height), _phi0(phi0), _uspot(_g), _xspot_center(_g), _rng(1), _density_scale_factor(1.0)
{
  _Omega = 1.0/( std::pow(_rhoorbit/_g.mass(),1.5) + _g.ang_mom() );

  _rorbit = std::sqrt(_rhoorbit*_rhoorbit + _zorbit*_zorbit);
  _thetaorbit = atan2(_rhoorbit,_zorbit);


  _tmin=-1500;
  _tmax=500;
  int N = int(std::ceil((_tmax-_tmin)/_Dt));

  _density_scale = new double[N];
  _decay_rate = 0.25;
  _density_scale[0] = 0.0;
  for (int i=1; i<N; i++)
    _density_scale[i] = std::pow( 10.0,  (_sigld*_rng.rand()+_ldm) ) + _density_scale[i-1]*std::exp(-_decay_rate*_Dt);
}

ED_SphericalKeplerianStochasticSpot::~ED_SphericalKeplerianStochasticSpot()
{
  delete[] _density_scale;
}


};
