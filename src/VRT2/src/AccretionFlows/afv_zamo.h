/***************************************************/
/*** Defines the velocity for a ZAMO flow        ***/
/*                                                 */
/***************************************************/

#ifndef VRT2_AFV_ZAMO_H
#define VRT2_AFV_ZAMO_H

#include "accretion_flow_velocity.h"
#include "metric.h"
#include "kerr.h"
#include "fourvector.h"


namespace VRT2 {
class AFV_ZAMO : public AccretionFlowVelocity
{
 public:
  AFV_ZAMO(Metric& g);
  virtual ~AFV_ZAMO() {};

  // User defined density
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);
};

inline FourVector<double>& AFV_ZAMO::get_velocity(double, double r, double theta, double)
{
  _u.mkcov(-1.0,0.0,0.0,0.0);

  _u *= 1.0/std::sqrt( -(_u*_u) );

  return _u;
}
};
#endif
