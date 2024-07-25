#include "afv_keplerian_inflow.h"

namespace VRT2 {

AFV_KeplerianInflow::AFV_KeplerianInflow(Metric& g, double ri, double Pdot)
  : AccretionFlowVelocity(g), _ri(ri), _Pdot(Pdot)
{
  // Set local metric
  _g_local = new Kerr(g.mass(),g.ang_mom()/g.mass());

  // Get local position
  std::valarray<double> x = _g.local_position();
  _g.reset(0,_ri,0.5*M_PI,0);
  
  // Set angular momentum and energy cutoffs
  _u = get_KeplerianInflow_velocity(_ri);
  _Omegai = _u.con(3)/_u.con(0);

  // return metric to where it was
  _g.reset(x);
}

};
