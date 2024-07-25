/*************************************************/
/*** An Azimuthal Magnetic field at fixed beta ***/
/*                                               */
/* Assumes that (i) the ion number density is    */
/* the same as the electron number density       */
/* (i.e., proton fraction of unity), (ii) the    */
/* ion temperature is ~1/r.                      */
/*                                               */
/* NOTE THAT THIS EXPECTS THAT THE METRIC        */
/* AND OTHER SUPPLIED ITEMS HAVE BEEN RESET      */
/* TO THE CURRENT POSITION.                      */
/*                                               */
/*************************************************/

#ifndef VRT2_MF_AZIMUTHAL_BETA_H
#define VRT2_MF_AZIMUTHAL_BETA_H

#include "vrt2_constants.h"
#include "magnetic_field.h"
#include <cmath>
#include <stdio.h>
using namespace std;

namespace VRT2 {
class MF_AzimuthalBeta : public MagneticField
{
 public:
  MF_AzimuthalBeta(Metric& g, AccretionFlowVelocity& u, ElectronDensity& ne, double beta);
  virtual ~MF_AzimuthalBeta() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);

 protected:
  AccretionFlowVelocity& _u;
  ElectronDensity& _ne;
  double _beta;
};

};
#endif
