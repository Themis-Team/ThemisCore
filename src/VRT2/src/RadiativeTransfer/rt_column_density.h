/***************************************************/
/*                                                 */
/* Notes:                                          */
/*   Implements column density integral.           */
/*                                                 */
/***************************************************/

// Only include once
#ifndef VRT2_RT_COLUMN_DENSITY
#define VRT2_RT_COLUMN_DENSITY

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

namespace VRT2 {
class RT_ColumnDensity: public RadiativeTransfer
{
 public:
  // Constructor
  RT_ColumnDensity(Metric& g,
		   ElectronDensity& ne, AccretionFlowVelocity& u);
  RT_ColumnDensity(const double y[], Metric& g,
		   ElectronDensity& ne, AccretionFlowVelocity& u);
  RT_ColumnDensity(FourVector<double>& x, FourVector<double>& k, Metric& g,
		   ElectronDensity& ne, AccretionFlowVelocity& u);
  virtual ~RT_ColumnDensity() { };

  // RT Coeffs (must include dl/dlambda!)
  // emissivity
  virtual std::valarray<double>& IQUV_ems(const double dydx[]);

  // Stable step size limiter (returns the minimum of the current step or the stable step)
  virtual double stable_step_size(double h, const double y[], const double dydx[]);

 private:
  ElectronDensity& _ne;
  AccretionFlowVelocity& _u;

  // Computes dlambda/dl
  double dl_dlambda(const double dydx[]);
};

};
#endif
