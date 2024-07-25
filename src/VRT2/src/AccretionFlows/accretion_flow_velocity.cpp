#include "accretion_flow_velocity.h"

namespace VRT2 {

AccretionFlowVelocity::AccretionFlowVelocity(Metric& g)
  : _g(g), _u(_g)
{
}

};
