#include "ed_Falcke.h"

namespace VRT2 {

ED_Falcke::ED_Falcke(double density_scale, FalckeJetModel& jet)
  : _density_scale(density_scale), _jet(jet)
{
}

};
