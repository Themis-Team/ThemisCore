/*********************************************/
/*** Defines Interface for ElectronDensity ***/
/*                                           */
/* IN UNITS OF cm^{-3}                       */
/*                                           */
/* Note that only the function get_density   */
/* must be defined, the others will be       */
/* automatically determined.                 */
/*                                           */
/* NOTE THAT THIS EXPECTS THAT THE METRIC    */
/* AND OTHER SUPPLIED ITEMS HAVE BEEN RESET  */
/* TO THE CURRENT POSITION.                  */
/*                                           */
/*********************************************/

#ifndef VRT2_ELECTRON_DENSITY_H
#define VRT2_ELECTRON_DENSITY_H

#include "fourvector.h"

namespace VRT2 {
class ElectronDensity
{
 public:
  virtual ~ElectronDensity() {};

  // User defined density
  virtual double get_density(double t, double r, double theta, double phi)=0;

  // Natural access functions
  double operator()(double t, double r, double theta, double phi);
  double operator()(const double y[]);
  double operator()(FourVector<double>& x);
};

inline double ElectronDensity::operator()(double t, double r, double theta, double phi)
{
  return get_density(t,r,theta,phi);
}
inline double ElectronDensity::operator()(const double y[])
{
  return get_density(y[0],y[1],y[2],y[3]);
}
inline double ElectronDensity::operator()(FourVector<double>& x)
{
  return get_density(x.con(0),x.con(1),x.con(2),x.con(3));
}

};
#endif
