#include "t_Falcke.h"

namespace VRT2 {

T_Falcke::T_Falcke(double temperature_scale, FalckeJetModel& jet)
  : _temperature_scale(temperature_scale), _jet(jet)
{
}

};
