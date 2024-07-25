/***********************************************/
/*** Defines Interface for MagneticField     ***/
/*                                             */
/* IN UNITS OF Gauss                           */
/*                                             */
/* Note that only the function get_field       */
/* must be defined, the others will be         */
/* automatically determined.                   */
/*                                             */
/* NOTE THAT THIS EXPECTS THAT THE METRIC      */
/* AND OTHER SUPPLIED ITEMS HAVE BEEN RESET    */
/* TO THE CURRENT POSITION.                    */
/*                                             */
/***********************************************/

#ifndef VRT2_MAGNETIC_FIELD_H
#define VRT2_MAGNETIC_FIELD_H

#include "metric.h"
#include "fourvector.h"
#include "accretion_flow_velocity.h"
#include "electron_density.h"

namespace VRT2 {
class MagneticField
{
 public:
  MagneticField(Metric& g);
  virtual ~MagneticField() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi) = 0;
  
  // Natural access functions
  FourVector<double>& operator()(double t, double r, double theta, double phi);
  FourVector<double>& operator()(const double y[]);
  FourVector<double>& operator()(FourVector<double>& x);
 protected:
  Metric& _g;
  FourVector<double> _b;
};

inline FourVector<double>& MagneticField::operator()(double t, double r, double theta, double phi)
{
  return get_field_fourvector(t,r,theta,phi);
}
inline FourVector<double>& MagneticField::operator()(const double y[])
{
  return get_field_fourvector(y[0],y[1],y[2],y[3]);
}
inline FourVector<double>& MagneticField::operator()(FourVector<double>& x)
{
  return get_field_fourvector(x.con(0),x.con(1),x.con(2),x.con(3));
}
};
#endif
