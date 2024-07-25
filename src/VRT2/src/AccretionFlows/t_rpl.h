/*************************************************************/
/*** Generates power-law temperature with                  ***/
/*                                                           */
/*  Te = temperature_scale * r^power * exp(-z^2/2(h*rho)^2)  */
/*************************************************************/

#ifndef VRT2_T_RPL_H
#define VRT2_T_RPL_H

#include "temperature.h"

namespace VRT2 {
class T_RPL : public Temperature
{
 public:
  T_RPL(double temperature_scale, double power, double h);
  virtual ~T_RPL() {};

  virtual double get_temperature(double t, double r, double theta, double phi);

 private:
  double _temperature_scale;
  double _power;
  //double _h;
};

inline double T_RPL::get_temperature(double,double r,double theta,double)
{
  //double z = r*std::cos(theta);
  //double rho = r*std::sin(theta) + 1.0e-10; // Tiny deals with z-axis
  return _temperature_scale * std::pow(r,_power); //std::pow(rho,_power) * std::exp( -z*z/(2.0*_h*_h*rho*rho) );
}
};
#endif
