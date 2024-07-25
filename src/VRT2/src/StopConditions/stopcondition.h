/**********************************************/
/****** Header file for stopcondition.cpp ******/
/* Notes:
   Gives conditions to stop evolution
*/
/**********************************************/

// Only include once
#ifndef VRT2_STOPCONDITION_H
#define VRT2_STOPCONDITION_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>

// Local Headers
#include "metric.h"
#include "fourvector.h"

namespace VRT2 {
class StopCondition{
 public:
  StopCondition(Metric& g, double router, double rinner) : _g(g), _rout(router), _rin(rinner), _omega_scale(1.0) {};
  virtual ~StopCondition() {};

  // These take y[] and dydx[] as arguments

  // The condition (1 stop, 0 don't stop)
  virtual int stop_condition(double[],double[]);

  // The intensity when stopped
  virtual double I(double[],int);

  // The Stokes' Parameters when stopped
  virtual std::valarray<double> IQUV(double[]);

  // Set frequency scale
  virtual void set_frequency_scale(double omega0) { _omega_scale = omega0; };

 protected:
  Metric& _g;
  const double _rout, _rin;
  double _omega_scale;
};
};
#endif
