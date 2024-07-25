/************************************************************/
/*** A Magnetic field at fixed beta and fixed radial pitch angle ***/
/*                                                          */
/* Assumes that (i) the ion number density is               */
/* the same as the electron number density                  */
/* (i.e., proton fraction of unity), (ii) the               */
/* ion temperature is ~1/r.                                 */
/*                                                          */
/* NOTE THAT THIS EXPECTS THAT THE METRIC                   */
/* AND OTHER SUPPLIED ITEMS HAVE BEEN RESET                 */
/* TO THE CURRENT POSITION.                                 */
/*                                                          */
/************************************************************/

#ifndef VRT2_MF_CONSTANT_RADIAL_PITCH_ANGLE_BETA_H
#define VRT2_MF_CONSTANT_RADIAL_PITCH_ANGLE_BETA_H

#include "vrt2_constants.h"
#include "magnetic_field.h"
#include <math.h>
#include <cmath>

namespace VRT2 {
class MF_ConstantRadialPitchAngleBeta : public MagneticField
{
 public:
  MF_ConstantRadialPitchAngleBeta(Metric& g, AccretionFlowVelocity& u, ElectronDensity& ne, double angle, double beta);
  virtual ~MF_ConstantRadialPitchAngleBeta() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);

 protected:
  AccretionFlowVelocity& _u;
  ElectronDensity& _ne;
  double _sa, _ca, _beta;
};
};
#endif
