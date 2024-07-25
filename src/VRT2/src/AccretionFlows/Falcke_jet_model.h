/*************************************************************************/
/***  Returns various parameters associated with the Falcke Jet model. ***/
/*************************************************************************/

#ifndef VRT2_FALCKE_JET_MODEL_H
#define VRT2_FALCKE_JET_MODEL_H


#include <algorithm>
#include <cmath>
#include <math.h>
using namespace std;

namespace VRT2 {
class FalckeJetModel
{
 public:
  FalckeJetModel(double Znozzle, double adiabatic_index);
  ~FalckeJetModel() {};

  double nozzle_height() { return _Znozzle; };
  double adiabatic_index() { return _adiabatic_index; };
  double gbs() { return _gbs; };

  double gbj(double z);   // gamma * beta along z-axis of jet
  double mach(double z);  // Mach number of jet
  double rperp(double z); // perpendicular size of jet


 private:
  double _Znozzle, _adiabatic_index;
  double _bs, _gbs;
};


};
#endif
