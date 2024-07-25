/****************************************************/
/*** Interface for temperature class              ***/
/****************************************************/

#ifndef VRT2_TEMPERATURE_H
#define VRT2_TEMPERATURE_H

#include "fourvector.h"

namespace VRT2 {
class Temperature
{
 public:
  virtual ~Temperature() {};

  // User defined density
  virtual double get_temperature(double t, double r, double theta, double phi)=0;

  // Natural access functions
  double operator()(double t, double r, double theta, double phi);
  double operator()(const double y[]);
  double operator()(FourVector<double>& x);
};

inline double Temperature::operator()(double t, double r, double theta, double phi)
{
  return get_temperature(t,r,theta,phi);
}
inline double Temperature::operator()(const double y[])
{
  return get_temperature(y[0],y[1],y[2],y[3]);
}
inline double Temperature::operator()(FourVector<double>& x)
{
  return get_temperature(x.con(0),x.con(1),x.con(2),x.con(3));
}
};
#endif
