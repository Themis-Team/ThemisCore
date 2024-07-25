#include "afv_shearing_inflow.h"

namespace VRT2 {

AFV_ShearingInflow::AFV_ShearingInflow(Metric& g, double ri, double infallRate, double subKep)
  : AccretionFlowVelocity(g), _ri(ri), _infallRate(infallRate), _subKep(subKep)
{
  // Set local metric
  _g_local = new Kerr(g.mass(),g.ang_mom()/g.mass());

}

};
