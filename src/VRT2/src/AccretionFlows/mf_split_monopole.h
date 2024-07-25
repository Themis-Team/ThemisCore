/************************************************/
/*** An Split Monopole Magnetic Field         ***/
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

#ifndef VRT2_MF_SPLIT_MONOPOLE_H
#define VRT2_MF_SPLIT_MONOPOLE_H

#include "vrt2_constants.h"
#include "magnetic_field.h"

namespace VRT2 {
class MF_SplitMonopole : public MagneticField
{
 public:
  MF_SplitMonopole(Metric& g, AccretionFlowVelocity& u, double B0);
  virtual ~MF_SplitMonopole() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);

 protected:
  AccretionFlowVelocity& _u;
  double _B0;
};
};
#endif
