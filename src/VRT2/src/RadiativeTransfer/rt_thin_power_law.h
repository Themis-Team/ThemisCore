/***************************************************/
/*                                                 */
/* Notes:                                          */
/*   Implements optically thin power-law spectrum. */
/*                                                 */
/***************************************************/

// Only include once
#ifndef VRT2_RT_THIN_POWER_LAW_H
#define VRT2_RT_THIN_POWER_LAW_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <vector>
#include <valarray>
#include <complex>

// Special Headers
#include "metric.h"
#include "fourvector.h"
#include "radiativetransfer.h"
#include "electron_density.h"
#include "accretion_flow_velocity.h"
#include "vrt2_constants.h"

namespace VRT2 {
class RT_ThinPowerLaw: public RadiativeTransfer
{
 public:
  // Constructor
  RT_ThinPowerLaw(Metric& g,
		  ElectronDensity& ne, AccretionFlowVelocity& u,
		  double spectral_index);
  RT_ThinPowerLaw(const double y[], Metric& g,
		  ElectronDensity& ne, AccretionFlowVelocity& u,
		  double spectral_index);
  RT_ThinPowerLaw(FourVector<double>& x, FourVector<double>& k, Metric& g,
		  ElectronDensity& ne, AccretionFlowVelocity& u,
		  double spectral_index);
  virtual ~RT_ThinPowerLaw() { };

  // RT Coeffs (must include dl/dlambda!)
  // emissivity
  virtual std::valarray<double>& IQUV_ems(const double dydx[]);

  // Stable step size limiter (returns the minimum of the current step or the stable step)
  virtual double stable_step_size(double h, const double y[], const double dydx[]);

 private:
  ElectronDensity& _ne;
  AccretionFlowVelocity& _u;
  double _spectral_index;

  // Computes dlambda/dl
  double dl_dlambda(const double dydx[]);
};

};
#endif
