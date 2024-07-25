#include "afv_subkeplerian.h"

namespace VRT2 {

AFV_SubKeplerian::AFV_SubKeplerian(Metric& g, double ri, double subkeplerian_factor)
  : AccretionFlowVelocity(g), _ri(ri), _subkeplerian_factor(subkeplerian_factor)
{
  // Set local metric
  _g_local = new Kerr(g.mass(),g.ang_mom()/g.mass());

  // Get local position
  std::valarray<double> x = _g.local_position();
  _g.reset(0,_ri,0.5*M_PI,0);

  // Set angular momentum and energy cutoffs
  //_u = get_keplerian_velocity(_ri);
  //_Omegai = _u.con(3)/_u.con(0);

  // return metric to where it was
  _g.reset(x);
}

}
