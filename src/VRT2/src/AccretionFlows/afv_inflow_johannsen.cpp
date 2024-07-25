#include "afv_inflow_johannsen.h"

namespace VRT2 {

AFV_InflowJohannsen::AFV_InflowJohannsen(Johannsen& g, double ri, double infallRate, double subKep)
  : AccretionFlowVelocity(g), _ri(ri), _infallRate(infallRate), _subKep(subKep), _uK(g)
{
  // Set local metric
  _g_local = new Johannsen(g.mass(),g.ang_mom()/g.mass(), g.alpha13(), g.alpha22(), g.alpha52(), g.epsilon(), g.beta());

}

};
