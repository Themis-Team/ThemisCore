/*********************************************************/
/*** Generates power-law density with                  ***/
/*                                                       */
/*  ne = density_scale * r^power * exp(-z^2/2(h*rho)^2)  */
/*********************************************************/

#ifndef VRT2_ED_RPL_CUTOFF_H
#define VRT2_ED_RPL_CUTOFF_H

#include "electron_density.h"

namespace VRT2 {
class ED_RPL_Cutoff : public ElectronDensity
{
 public:
  ED_RPL_Cutoff(double density_scale, double power, double h, double rmin);
  virtual ~ED_RPL_Cutoff() {};

  virtual double get_density(double t, double r, double theta, double phi);

 private:
  double _density_scale;
  double _power;
  double _h;
  double _rmin;
};

inline double ED_RPL_Cutoff::get_density(double,double r,double theta,double)
{
  if (r>_rmin)
  {
    double z = r*std::cos(theta);
    double rho = r*std::sin(theta) + 1.0e-10; // Tiny deals with z-axis
    
    return _density_scale * std::pow(rho,_power) * std::exp( -z*z/(2.0*_h*_h*rho*rho) );
  }
  return _density_scale*1e-20;
}
};
#endif
