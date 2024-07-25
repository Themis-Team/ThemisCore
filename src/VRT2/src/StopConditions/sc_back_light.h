/**********************************************/
/****** Header file for sc_back_light.cpp ******/
/* Notes:
   Gives conditions to stop evolution
*/
/**********************************************/

// Only include once
#ifndef VRT2_SC_BACKLIGHT_H
#define VRT2_SC_BACKLIGHT_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>

// Local Headers
#include "metric.h"
#include "fourvector.h"
#include "stopcondition.h"

namespace VRT2 {
class SC_BackLight : public StopCondition
{
 public:
 SC_BackLight(Metric& g, double router, double rinner, double Iinf) : StopCondition(g,router,rinner), _Iinf(Iinf) {};
  virtual ~SC_BackLight() {};

  // These take y[] and dydx[] as arguments

  // The condition (1 stop, 0 don't stop)
  virtual int stop_condition(double[],double[]);

  // The intensity when stopped
  virtual double I(double[],int);

  // The Stokes' Parameters when stopped
  virtual std::valarray<double> IQUV(double[]);

 protected:
  double _Iinf;
};
};
#endif
