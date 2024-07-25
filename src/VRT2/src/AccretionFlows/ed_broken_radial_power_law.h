/*********************************************************/
/*** Generates power-law density with                  ***/
/*                                                       */
/*  ne = density_scale * r^power * exp(-z^2/2(h*rho)^2)  */
/*********************************************************/

#ifndef VRT2_ED_BRPL_H
#define VRT2_ED_BRPL_H

#include "electron_density.h"
#include <cmath>

namespace VRT2 {
class ED_BRPL : public ElectronDensity
{
 public:
  ED_BRPL(double density_scale, double power1, double power2, double rbreak, double h);
  virtual ~ED_BRPL() {};

  virtual double get_density(double t, double r, double theta, double phi);

 private:
  double _density_scale;
  double _power1, _power2, _rbreak;
  double _h;
};
inline double ED_BRPL::get_density(double t,double r,double theta,double phi)
{
  double z = r*std::cos(theta);
  double rho = std::fabs(r*std::sin(theta)) + 1.0e-10; // Tiny deals with z-axis

  if (rho>_rbreak)
    return _density_scale * std::pow(rho,_power1) * std::exp( -z*z/(2.0*_h*_h*rho*rho) );
  else
    return _density_scale * std::pow(rho/_rbreak,_power2)*std::pow(_rbreak,_power1) * std::exp( -z*z/(2.0*_h*_h*rho*rho) );
}
};
#endif
