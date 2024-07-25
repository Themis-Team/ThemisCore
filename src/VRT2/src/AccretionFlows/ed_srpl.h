/*********************************************************/
/*** Generates power-law density with                  ***/
/*                                                       */
/*  ne = density_scale * r^power                         */
/*********************************************************/

#ifndef VRT2_ED_SRPL_H
#define VRT2_ED_SRPL_H

#include "electron_density.h"

namespace VRT2 {
class ED_SRPL : public ElectronDensity
{
 public:
  ED_SRPL(double density_scale, double power);
  virtual ~ED_SRPL() {};

  virtual double get_density(double t, double r, double theta, double phi);

 private:
  double _density_scale;
  double _power;
};

inline double ED_SRPL::get_density(double,double r,double,double)
{
  return _density_scale * std::pow(r,_power);                                      ;
}
};
#endif
