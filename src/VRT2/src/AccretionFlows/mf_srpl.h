/************************************************/
/*** An Radial Power-law Radial Field         ***/
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

#ifndef VRT2_MF_SRPL_H
#define VRT2_MF_SRPL_H

#include "vrt2_constants.h"
#include "magnetic_field.h"

namespace VRT2 {
class MF_SRPL : public MagneticField
{
 public:
  MF_SRPL(Metric& g, double B0, double power);
  virtual ~MF_SRPL() {};
  
  // User defined field
  virtual FourVector<double>& get_field_fourvector(double t, double r, double theta, double phi);

 protected:
  double _B0, _power;
};

};
#endif
