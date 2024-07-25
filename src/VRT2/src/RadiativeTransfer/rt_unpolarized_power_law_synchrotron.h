/**********************************************************/
/*** Implements polarized power-law syncrotron emission ***/
/*                                                        */
/* Notes:                                                 */
/*   Implements synchrotron emission fromm power- law     */
/* electron distribution.                                 */
/*                                                        */
/* Jones & Odell, 1977 ApJ, 214, 522                      */
/*                                                        */
/**********************************************************/

// Only include once
#ifndef VRT2_RT_UNPOLARIZED_POWER_LAW_SYNCHROTRON_H
#define VRT2_RT_UNPOLARIZED_POWER_LAW_SYNCHROTRON_H

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
#include "vrt2_constants.h"

namespace VRT2 {
class RT_UnpolarizedPowerLawSynchrotron : public RadiativeTransfer
{
 public:
  // Constructor
  RT_UnpolarizedPowerLawSynchrotron(Metric& g,
			 ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
			 double spectral_index, double gamma_min);
  RT_UnpolarizedPowerLawSynchrotron(const double y[], Metric& g,
			 ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
			 double spectral_index, double gamma_min);
  RT_UnpolarizedPowerLawSynchrotron(FourVector<double>& x, FourVector<double>& k, Metric& g,
			 ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
			 double spectral_index, double gamma_min);
  virtual ~RT_UnpolarizedPowerLawSynchrotron() { };

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
  AccretionFlowVelocity& _u;
  MagneticField& _B;

  double _spectral_index;
  double _gamma_min;
  double _Cjnu, _Calphanu, _epsilon_Q, _zeta_Q; // Constants that can be predefined
  double _ComegaB, _Cn; // More constants that can be predefined

  double _omega, _omegaB, _n0; // Some common values among all return functions
  double _sn_alpha; // Sine of angle between field and k
  double _sn, _cs; // sine and cosine of the angle between the field defined Stokes basis and the fiducial Stokes basis

  // Set constants only once
  void set_constants();

  // Get common functions
  void set_common_funcs();

  // Get rotation angle to align with z-aligned Stokes basis
  void get_Stokes_alignment_angle(FourVector<double>& u, FourVector<double>& b, double& cs, double& sn);

  // Computes dlambda/dl
  double dl_dlambda(const double dydx[]);

  // Gamma function for constants
  double gammln(double x);
};

};
#endif
