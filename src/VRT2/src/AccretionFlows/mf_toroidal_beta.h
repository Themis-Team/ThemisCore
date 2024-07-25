/************************************************/
/*** An Toroidal Magnetic field at fixed beta ***/
/*                                              */
/* Assumes that (i) the ion number density is   */
/* the same as the electron number density      */
/* (i.e., proton fraction of unity), (ii) the   */
/* ion temperature is ~1/r.                     */
/*                                              */
/* NOTE THAT THIS EXPECTS THAT THE METRIC       */
/* AND OTHER SUPPLIED ITEMS HAVE BEEN RESET     */
/* TO THE CURRENT POSITION.                     */
/*                                              */
/************************************************/

#ifndef VRT2_MF_TOROIDAL_BETA_H
#define VRT2_MF_TOROIDAL_BETA_H

#include "vrt2_constants.h"
#include "magnetic_field.h"
#include "vrt2_globs.h"

namespace VRT2 {
class MF_ToroidalBeta : public MagneticField
{
 public:
  MF_ToroidalBeta(Metric& g, AccretionFlowVelocity& u, ElectronDensity& ne, double beta);
  virtual ~MF_ToroidalBeta() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);

 protected:
  AccretionFlowVelocity& _u;
  ElectronDensity& _ne;
  double _beta;
};

};
#endif
