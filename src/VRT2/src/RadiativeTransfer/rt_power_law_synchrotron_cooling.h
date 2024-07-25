/**********************************************************/
/*** Implements polarized power-law syncrotron emission ***/
/*   with a exponential cooling break.                    */
/* Notes:                                                 */
/*   Implements synchrotron emission fromm power- law     */
/* electron distribution, which an exponential cut-off    */
/* that models cooling.                                   */
/* Jones & Odell, 1977 ApJ, 214, 522                      */
/*                                                        */
/**********************************************************/

// Only include once
#ifndef VRT2_RT_POWER_LAW_SYNCHROTRON_COOLING_H
#define VRT2_RT_POWER_LAW_SYNCHROTRON_COOLING_H

// Standard Library Headers
#include <cmath>
#include <math.h>
using namespace std;
#include <vector>
#include <valarray>
#include <complex>
#include <limits>

// Special Headers
#include "metric.h"
#include "fourvector.h"
#include "radiativetransfer.h"
#include "electron_density.h"
#include "accretion_flow_velocity.h"
#include "magnetic_field.h"
#include "vrt2_globs.h"
#include "vrt2_constants.h"

namespace VRT2 {
class RT_PowerLawSynchrotronCooling : public RadiativeTransfer
{
 public:
  // Constructor
  RT_PowerLawSynchrotronCooling(Metric& g,
			 ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
			 double injection_time, double spectral_index, double gamma_min);
  RT_PowerLawSynchrotronCooling(const double y[], Metric& g,
			 ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
			 double injection_time, double spectral_index, double gamma_min);
  RT_PowerLawSynchrotronCooling(FourVector<double>& x, FourVector<double>& k, Metric& g,
			 ElectronDensity& ne, AccretionFlowVelocity& u, MagneticField& B,
			 double injection_time, double spectral_index, double gamma_min);
  virtual ~RT_PowerLawSynchrotronCooling() { };

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



  virtual void dump(std::ostream& dout, double dydx[]);

 private:
  ElectronDensity& _ne;
  AccretionFlowVelocity& _u;
  MagneticField& _B;

  const double _injection_time; //in M;
  double _delta_t; //Change since injection time in M
  double _spectral_index;
  double _gamma_min;
  double _gamma_min_time, _gamma_max_time; //Maximum gamma min/max after time evolution
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

