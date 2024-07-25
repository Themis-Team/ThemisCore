/***************************************************/
/*** Defines Interface for AccretionFlowVelocity ***/
/*                                                 */
/* IN UNITS OF c                                   */
/*                                                 */
/* Note that only the function get_velocity        */
/* must be defined, the others will be             */
/* automatically determined.                       */
/*                                                 */
/* NOTE THAT THIS EXPECTS THAT THE METRIC          */
/* AND OTHER SUPPLIED ITEMS HAVE BEEN RESET        */
/* TO THE CURRENT POSITION.                        */
/*                                                 */
/***************************************************/

#ifndef VRT2_ACCRETION_FLOW_VELOCITY_H
#define VRT2_ACCRETION_FLOW_VELOCITY_H

#include "metric.h"
#include "fourvector.h"

namespace VRT2 {

class AccretionFlowVelocity
{
 public:
  AccretionFlowVelocity(Metric& g);
  virtual ~AccretionFlowVelocity() {};

  // User defined velocity fourvector
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi) = 0;

  // Natural access functions
  FourVector<double>& operator()(double t, double r, double theta, double phi);
  FourVector<double>& operator()(const double y[]);
  FourVector<double>& operator()(FourVector<double>& x);
  
 protected:
  Metric& _g;
  FourVector<double> _u;

};

inline FourVector<double>& AccretionFlowVelocity::operator()(double t, double r, double theta, double phi)
{
  return get_velocity(t,r,theta,phi);
}
inline FourVector<double>& AccretionFlowVelocity::operator()(const double y[])
{
  return get_velocity(y[0],y[1],y[2],y[3]);
}
inline FourVector<double>& AccretionFlowVelocity::operator()(FourVector<double>& x)
{
  return get_velocity(x.con(0),x.con(1),x.con(2),x.con(3));
}

};
#endif
