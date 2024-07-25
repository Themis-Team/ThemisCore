#include "afv_Falcke.h"

namespace VRT2 {

AFV_Falcke::AFV_Falcke(Metric& g, FalckeJetModel& jet)
  : AccretionFlowVelocity(g), _jet(jet)
{
  // Set rmin to marginally bound orbit (from Bardeen, 1972)
  _rmin = ( 2.0*_g.mass() - _g.ang_mom() + 2.0*std::sqrt(_g.mass())*std::sqrt(_g.mass()-_g.ang_mom()) )/_g.mass();
  //_rmin = 6.0;
}

};
