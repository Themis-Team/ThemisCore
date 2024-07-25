/***************************************************/
/****** Header file for rt_pppa.cpp           ******/
/* Notes:                                          */
/*   Implements Parallel-propagation method for    */
/* determining the rotation angle of the           */
/* polarization basis relative to the local        */
/* vertical.                                       */
/***************************************************/

// Only include once
#ifndef VRT2_RT_PP_PA_H
#define VRT2_RT_PP_PA_H

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

namespace VRT2 {
class RT_PP_PA : public RadiativeTransfer
{
 public:
  // Constructor
  RT_PP_PA(Metric& g);
  RT_PP_PA(const double y[], Metric& g);
  RT_PP_PA(FourVector<double>& x,FourVector<double>& k, Metric& g);
  virtual ~RT_PP_PA() { };

  virtual void IQUV_rotate(double iquv[], double xi, const double yi[], const double dydxi[], double xf, const double yf[], const double dydxf[]);

  // Construct an array of position angles that we interpolate off of.
  virtual void IQUV_integrate_initialize(std::vector<double> ya[], std::vector<double> dydxa[], std::valarray<double>& iquv0);


 private:
  // Tables of position angle
  std::vector<double> _lt, _pat;

  // Get dpa/dlambda
  double delta_tPA(double lambdai, double lambdaf);
  double PP_polarization_angle(double lambda);

  // Local pointers to ray positions
  std::vector<double> *_ya, *_dydxa;

  // Linear interpolation to get x,k and dx,dk
  inline void PPPAinterp(double lambda, double y[], double dydx[]);

  // Numerics for integration
  void PPPAderivs(double,const double[],double[]);
  int PPPArkqs(double[],double[],int,double&,double,double,double[],double&,double&);
  void PPPArkck(double [],double [],int,double,double,double [],double []);
};

};
#endif
