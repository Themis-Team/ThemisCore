/************************************************************/
/*** A Magnetic field at fixed beta and fixed pitch angles***/
/*   in the radial and vertical direction                   */
/*                                                          */
/* First specify the angle given by B.phi and then the      */
/* angle defining the rotation around phi with vertical     */
/* being zero.                                              */
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

#ifndef VRT2_MF_CONSTANT_PITCH_ANGLES_BETA_H
#define VRT2_MF_CONSTANT_PITCH_ANGLES_BETA_H

#include "vrt2_constants.h"
#include "magnetic_field.h"
#include "accretion_flow_velocity.h"
#include "electron_density.h"

namespace VRT2 {
class MF_ConstantPitchAnglesBeta : public MagneticField
{
 public:
  MF_ConstantPitchAnglesBeta(Metric& g, AccretionFlowVelocity& u, ElectronDensity& ne, double angleBphi, double angleZ, double beta);
  virtual ~MF_ConstantPitchAnglesBeta() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);

 protected:
  AccretionFlowVelocity& _u;
  ElectronDensity& _ne;
  double _sa1, _ca1, _sa2, _ca2, _beta;
};

};
#endif
