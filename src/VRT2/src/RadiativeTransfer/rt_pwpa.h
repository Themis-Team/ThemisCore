/***************************************************/
/****** Header file for rt_pwpa.cpp           ******/
/* Notes:                                          */
/*   Implements Penrose-Walker polarisation angle  */
/* rotation.                                       */
/***************************************************/

// Only include once
#ifndef VRT2_RT_PW_PA_H
#define VRT2_RT_PW_PA_H

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
class RT_PW_PA : public RadiativeTransfer
{
 public:
  // Constructor
  RT_PW_PA(Metric& g, double Theta);
  RT_PW_PA(const double y[], Metric& g, double Theta);
  RT_PW_PA(FourVector<double>& x,FourVector<double>& k, Metric& g, double Theta);
  virtual ~RT_PW_PA() { };

  virtual void IQUV_rotate(double iquv[], double xi, const double yi[], const double dydxi[], double xf, const double yf[], const double dydxf[]);

  // Characteristic local length to affine parameter difference
  virtual double dlambda(const double y[], const double dydx[]);

  void set_theta(double Theta) { _Theta = Theta; };

  private:
  // Angle at infinity of the ray
  double _Theta;

  // Get polarization angle at this position for the vertical direction via Penrose-Walker constant
  double PW_polarization_angle(const double y[]);
  
  // Get dpa/dlambda
  double delta_tPA(const double yi[], const double yf[]);

};

};
#endif
