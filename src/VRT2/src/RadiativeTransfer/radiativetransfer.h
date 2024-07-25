/***************************************************/
/****** Header file for radiativetransfer.cpp ******/
/* Notes:                                          */
/*   Describes interface for RT.                   */
/*   Expects that Stokes parameters are aligned    */
/*    with the time measured by the ZAMO observer. */
/***************************************************/

// Only include once
#ifndef VRT2_RADIATIVETRANSFER_H
#define VRT2_RADIATIVETRANSFER_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <vector>
#include <valarray>

#include <iostream>
#include <iomanip>

// Special Headers
#include "metric.h"
#include "fourvector.h"

namespace VRT2 {
class RadiativeTransfer
{
 public:
  // Constructor
  RadiativeTransfer(Metric& g);
  RadiativeTransfer(const double y[], Metric& g);
  RadiativeTransfer(FourVector<double>& x,FourVector<double>& k, Metric& g);
  virtual ~RadiativeTransfer() { };

  // Set frequency scale
  virtual void set_frequency_scale(double omega0) { _omega_scale = omega0; };

  // Set length scale
  virtual void set_length_scale(double L) { _length_scale = L; };

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
  virtual void IQUV_rotate(double iquv[], double lambdai, const double yi[], const double dydxi[], double lambdaf, const double yf[], const double dydxf[]);

  // Integrated Stokes parameters
  virtual std::valarray<double>& IQUV_integrate(std::vector<double> y[],std::vector<double> dydx[],std::valarray<double>& iquv0);

  // Any additional initialization that must be done prior to the IQUV integration.
  virtual void IQUV_integrate_initialize(std::vector<double> y[], std::vector<double> dydx[], std::valarray<double>& iquv0) {};

  
  virtual void dump(std::ostream& dout, double dydx[]) {};

 protected:
  Metric& _g; // Metric
  FourVector<double> _x, _k; // local x and k
  double _omega_scale; // Scaling of omega ( _omega_scale*k is the wave vector in real units )
  double _length_scale; // Scaling of dl

  std::valarray<double> _iquv_ems;
  std::valarray<double> _iquv_abs;
};

};
#endif
