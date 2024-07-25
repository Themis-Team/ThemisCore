/*********************************************************/
/*** Generates power-law density with                  ***/
/*                                                       */
/*  ne = density_scale * r^power * exp(-z^2/2(h*rho)^2)  */
/*********************************************************/

#ifndef VRT2_ED_RPL_H
#define VRT2_ED_RPL_H

#include "electron_density.h"
#include "fast_math.h"
#include "vrt2_globs.h"

namespace VRT2 {
class ED_RPL : public ElectronDensity
{
 public:
  ED_RPL(double density_scale, double power, double h);
  virtual ~ED_RPL() {};

  virtual double get_density(double t, double r, double theta, double phi);

 private:
  double _density_scale;
  double _power;
  double _h;
};

inline double ED_RPL::get_density(double,double r,double theta,double)
{
  double z = r*std::cos(theta);
  double rho = std::fabs(r*std::sin(theta)) + 1.0e-10; // Tiny deals with z-axis

  /*
  //double d = _density_scale * std::pow(r,_power) * std::exp( -z*z/(2.0*_h*_h*rho*rho) );
  double d = _density_scale * FastMath::pow(r,_power) * FastMath::exp( -z*z/(2.0*_h*_h*rho*rho) );

  if (vrt2_isnan(d))
    std::cout << "ED_RPL:"
	      << std::setw(15) << FastMath::pow(r,_power)
	      << std::setw(15) << FastMath::exp( -z*z/(2.0*_h*_h*rho*rho) )
	      << std::endl;

  return d;
  */
  //std::cout << "ed_rpl.cpp: " <<   _density_scale * FastMath::pow(r,_power) * FastMath::exp( -z*z/(2.0*_h*_h*rho*rho) ) 
  //<< std::endl;
  return _density_scale * FastMath::pow(r,_power) * FastMath::exp( -z*z/(2.0*_h*_h*rho*rho) );
  
  //return _density_scale * std::pow(rho,_power) * std::exp( -z*z/(2.0*_h*_h*rho*rho) );

}
};
#endif
