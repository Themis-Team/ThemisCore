/***************************************************/
/*                                                 */
/* Notes:                                          */
/*   Determines maximum HT-error along a geodesic. */
/*                                                 */
/***************************************************/

// Only include once
#ifndef VRT2_RT_HARTLE_THORNE_CHECK
#define VRT2_RT_HARTLE_THORNE_CHECK

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <vector>
#include <valarray>
#include <complex>

// Special Headers
#include "metric.h"
#include "kerr.h"
#include "fourvector.h"
#include "radiativetransfer.h"

namespace VRT2 {
class RT_HartleThorneCheck: public RadiativeTransfer
{
 public:
  // Constructor
  RT_HartleThorneCheck(Metric& g, double M, double a);
  RT_HartleThorneCheck(const double y[], Metric& g, double M, double a);
  RT_HartleThorneCheck(FourVector<double>& x, FourVector<double>& k, Metric& g, double M, double a);
  virtual ~RT_HartleThorneCheck() { };


 // Integrated Stokes parameters
  virtual std::valarray<double>& IQUV_integrate(std::vector<double> y[],std::vector<double> dydx[],std::valarray<double>& iquv0);

 private:
  Kerr _gk;

  double ht_fdiff(double y[]);
  
};
};
#endif
