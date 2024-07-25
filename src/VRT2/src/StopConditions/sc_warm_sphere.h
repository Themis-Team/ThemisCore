/**********************************************/
/****** Header file for stopcondition.cpp ******/
/* Notes:
   Gives conditions to stop evolution
*/
/**********************************************/

// Only include once
#ifndef VRT2_SCWARM_SPHERE_H
#define VRT2_SCWARM_SPHERE_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>

// Local Headers
#include "metric.h"
#include "fourvector.h"
#include "vrt2_constants.h"
#include "stopcondition.h"

namespace VRT2 {
class SCWarmSphere : public StopCondition
{
 public:
  SCWarmSphere(Metric& g, double router, double rinner, double T);
  virtual ~SCWarmSphere() {};

  // These take y[] and dydx[] as arguments

  // The condition (1 stop, 0 don't stop)
  virtual int stop_condition(double[],double[]);

  // The intensity when stopped
  virtual double I(double[],int);

  // The Stokes' Parameters when stopped
  virtual std::valarray<double> IQUV(double[]);

  // Some access functions
  virtual void set_inner_radius(double rinner) { _rin=rinner; };
  virtual void set_temperature(double T) { _T=VRT2_Constants::k*T; };

 protected:
  Metric& _g;
  double _rout, _rin, _T;
};

class SCAccretingWarmSphere : public SCWarmSphere
{
 public:
  SCAccretingWarmSphere(Metric& g, double router, double rinner, double Mdot);
  virtual ~SCAccretingWarmSphere() {};

  virtual void set_inner_radius(double rinner);

  double T_from_Mdot(double Mdot, double rinner);

 protected:
  double _Mdot;
};
};
#endif
