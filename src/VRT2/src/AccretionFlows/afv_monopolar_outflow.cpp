#include "afv_monopolar_outflow.h"

namespace VRT2 {

AFV_MonopolarOutflow::AFV_MonopolarOutflow(Metric& g, double e, double r0, double Omega)
  : AccretionFlowVelocity(g), _e(e), _r0(r0), _Omega(Omega)
{
  _sgn = (_Omega<0 ? -1 : 1);

  // Set local metric
  _g_local = new Kerr(g.mass(),g.ang_mom()/g.mass());
}
};
