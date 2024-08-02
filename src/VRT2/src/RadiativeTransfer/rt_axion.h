/**********************************************************/
/*** Implements axion birefringence on emission         ***/
/*                                                        */
/* Notes:                                                 */
/*   Implements axion birefringence for superradiant      */
/*   axion clouds around black holes.                     */
/*                                                        */
/*                                                        */
/* Chen et al. JCAP                                       */
/*                                                        */
/**********************************************************/

// Only include once
#ifndef VRT2_RT_AXION_H
#define VRT2_RT_AXION_H

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
#include "accretion_flow_velocity.h"
#include "vrt2_globs.h"
#include "vrt2_constants.h"

namespace VRT2 {
class RT_Axion : public RadiativeTransfer
{
 public:
  // Constructor
  RT_Axion(Metric& g,
	   AccretionFlowVelocity& u,
	   double dn, double ma, double ga,
	   int n=1,int l=2, int m=1);
  RT_Axion(const double y[], Metric& g,
	   AccretionFlowVelocity& u,
	   double dn, double ma, double ga, 
	   int n=1,int l=2, int m=1);
  RT_Axion(FourVector<double>& x, FourVector<double>& k, Metric& g,
	   AccretionFlowVelocity& u,
	   double dn, double ma, double ga, 
	   int n=1,int l=2, int m=1);
  virtual ~RT_Axion() { };

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
  AccretionFlowVelocity& _u;

  double _dn; // Density normalization for the axion cloud in TBD.
  double _ma; // Compton wavelength of the axion in eV
  double _ga; // Axion-photon coupling constant in TBD.

  int _n, _l, _m; // Quantum numbers for the axion cloud.

  double _omega; // Some common values among all return functions
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
};


};
#endif

