/***************************************************/
/****** Header file for rt_multi.cpp          ******/
/* Notes:                                          */
/*   Wrapper for multiple radiative transfers.     */
/***************************************************/

// Only include once
#ifndef VRT2_RT_MULTI_H
#define VRT2_RT_MULTI_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <vector>
#include <valarray>
#include <algorithm>

// Special Headers
#include "metric.h"
#include "fourvector.h"
#include "radiativetransfer.h"
#include "rt_rg.h"

namespace VRT2 {
class RT_Multi : public RT_RungeKutta
{
 public:
  // Constructor
  RT_Multi(Metric& g, std::vector<RadiativeTransfer*> rts);
  RT_Multi(const double y[], Metric& g, std::vector<RadiativeTransfer*> rts);
  RT_Multi(FourVector<double>& x,FourVector<double>& k, Metric& g, std::vector<RadiativeTransfer*> rts);
  virtual ~RT_Multi() { };

  // Set frequency scale
  virtual void set_frequency_scale(double omega0);

  // Set length scale
  virtual void set_length_scale(double L);

  // Reinitialize
  virtual void reinitialize(const double y[]);
  virtual void reinitialize(FourVector<double>& x, FourVector<double>& k);

  // Characteristic local length to affine parameter difference
  virtual double dlambda(const double y[], const double dydx[]);

  // RT Coeffs (must include dl/dlambda!)
  // absorptivity
  virtual std::valarray<double>& IQUV_abs(const double iquv[], const double dydx[]);
  // isotropic absorptivity for calculating optical depth
  virtual double isotropic_absorptivity(const double dydx[]);
  // emissivity
  virtual std::valarray<double>& IQUV_ems(const double dydx[]);

  // Linear RT Change
  virtual void IQUV_rotate(double iquv[], double xi, const double yi[], const double dydxi[], double xf, const double yf[], const double dydxf[]);

  // Stable step size limiter (returns the minimum of the current step or the stable step)
  virtual double stable_step_size(double h, const double y[], const double dydx[]);

  // Any additional initialization that must be done prior to the IQUV integration.
  virtual void IQUV_integrate_initialize(std::vector<double> y[], std::vector<double> dydx[], std::valarray<double>& iquv0);



 private:
  std::vector<RadiativeTransfer*> _rts;


 protected:
  virtual void dump_ray(std::string);
};

};
#endif
