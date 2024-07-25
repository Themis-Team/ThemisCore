/***********************************************************/
/***  Implements polarized thermal synchrotron emission  ***/
/*                                                         */
/* Notes:                                                  */
/*   Implements synchrotron emission from a relativistic   */
/* thermal electron distribution.                          */
/*                                                         */
/* Unpolarized emission from:                              */
/*   Yuan, Quataert, Narayan, 2003 ApJ, 598, 301           */
/* Polarization fraction from:                             */
/*   Petrosian & McTiernan, 1983 Phys. Fluids, 26, 3023    */
/*                                                         */
/***********************************************************/

// Only include once
#ifndef VRT2_RT_THERMAL_SYNCHROTRON_H
#define VRT2_RT_THERMAL_SYNCHROTRON_H

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
#include "magnetic_field.h"
#include "temperature.h"
#include "vrt2_constants.h"
#include "fast_math.h"
#include "vrt2_globs.h"

namespace VRT2 {
class RT_ThermalSynchrotron : public RadiativeTransfer
{
 public:
  // Constructor
  RT_ThermalSynchrotron(Metric& g,
			ElectronDensity& ne, Temperature& Te, AccretionFlowVelocity& u, MagneticField& B);
  RT_ThermalSynchrotron(const double y[], Metric& g,
			ElectronDensity& ne, Temperature& Te, AccretionFlowVelocity& u, MagneticField& B);
  RT_ThermalSynchrotron(FourVector<double>& x, FourVector<double>& k, Metric& g,
			ElectronDensity& ne, Temperature& Te, AccretionFlowVelocity& u, MagneticField& B);
  virtual ~RT_ThermalSynchrotron() { };

  // Set frequency scale
  virtual void set_frequency_scale(double omega0);

  // Set length scale
  virtual void set_length_scale(double L);

  // Reinitialize
  virtual void reinitialize(const double y[]);
  virtual void reinitialize(FourVector<double>& x, FourVector<double>& k);

  // Characteristic local length to affine parameter difference
  //virtual double dlambda(const double y[], const double dydx[]);

  // RT Coeffs (must include dl/dlambda!)
  // absorptivity
  virtual std::valarray<double>& IQUV_abs(const double iquv[], const double dydx[]);
  // isotropic absorptivity for calculating optical depth
  virtual double isotropic_absorptivity(const double dydx[]);
  // emissivity
  virtual std::valarray<double>& IQUV_ems(const double dydx[]);

 private:
  ElectronDensity& _ne;
  Temperature& _Te;
  AccretionFlowVelocity& _u;
  MagneticField& _B;

  double _Cjnu, _Calphanu, _ComegaB, _Cthetae; // Precomputed constants
  double _jnu; // emissivity / omega^3
  double _alphanu; // absorption coefficient
  double _epsilon_Q; // Polarization fraction
  double _cs, _sn; // Stokes alignment angles
  double _cs_alpha; // Agnle wrt magnetic field

  // Set constants only once
  void set_constants();

  // Get common functions
  void set_common_funcs();

  // Get rotation angle to align with z-aligned Stokes basis
  void get_Stokes_alignment_angle(FourVector<double>& u, FourVector<double>& b, double& cs, double& sn);

  // Computes dlambda/dl
  double dl_dlambda(const double dydx[]);

  // Generate a lookup table for K2
  void generate_K2_lookup_table();
  std::valarray<double> _x_K2, _K2_K2;
  double _x_K2_stp, _x_K2_min, _x_K2_max;
  double lnK2(double x);
  // Numerical recipes Bessel function numerics
  double bessi0(double x);
  double bessk0(double x);
  double bessi1(double x);
  double bessk1(double x);
  double bessk(int n, double x);
};

};
#endif




