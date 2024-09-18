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
   RT_Axion(Metric &g,
            AccretionFlowVelocity &u,
            double M, double ma, double ga);
   RT_Axion(const double y[], Metric &g,
            AccretionFlowVelocity &u,
            double M, double ma, double ga);
   RT_Axion(FourVector<double> &x, FourVector<double> &k, Metric &g,
            AccretionFlowVelocity &u,
            double M, double ma, double ga);
   virtual ~RT_Axion() {};

   // Set frequency scale
   virtual void set_frequency_scale(double omega0);

   // Set length scale
   virtual void set_length_scale(double L);

   // Reinitialize
   virtual void reinitialize(const double y[]);
   virtual void reinitialize(FourVector<double> &x, FourVector<double> &k);

   // Characteristic local length to affine parameter difference
   // virtual double dlambda(const double y[], const double dydx[]);

   // RT Coeffs (must include dl/dlambda!)
   // absorptivity
   virtual std::valarray<double> &IQUV_abs(const double iquv[], const double dydx[]);
   // isotropic absorptivity for calculating optical depth
   virtual double isotropic_absorptivity(const double dydx[]);
   // emissivity
   virtual std::valarray<double> &IQUV_ems(const double dydx[]);

   virtual void dump(std::ostream &dout, double dydx[]);


   // Axion Period Access
   double period() const { return 2.0*M_PI/_omega_21; };

 private:
  AccretionFlowVelocity& _u;

  double _M; // Mass of the black hole in solar masses
  double _ma; // Compton wavelength of the axion in eV
  double _ga; // Axion-photon coupling constant in TBD.

  double _omega; // Some common values among all return functions
  double _sn_alpha; // Sine of angle between field and k
  double _sn, _cs; // sine and cosine of the angle between the field defined Stokes basis and the fiducial Stokes basis

  // Set constants only once
  void set_constants();
  //std::tuple<double, double, double> set_constants(double alpha);

  double _alpha; // Fine structure constant
  double _mu; // ALP mass in Planck units
  // double _E21; // Energy of axion at n=2, l=1
  // double _beta; // Parameter beta to save times
  // double _omega_axion; // Eigenfrequency
  // double _alpha_term; // Alpha term to save time
  // double _pre_factor; // A prefactor A in the normalization factor
  // double _norm_factor; // Normalization factor for the axion cloud
  double _rp; // outer event horzion
  double _rm; // inner event horzion
  double _omega_crit; // Critical frequency of superradiance
  double _omega_21; // Eigenfrequency of axion at n=2, l=1
  double _sigma; // Dolan 2007 defined
  double _q; // same as above
  double _chi; // same as above
  double _Re_a_max; // the maximum value of Re(a) unnormalized
  // double _R_norm_factor; // Normalization factor for the radial function
  double _norm_factor; // totoal normalization factor for the axion field

  // Get common functions
  void set_common_funcs();

  // double beta_r; // Dimensionless r
  double _R1; // common radial part
  double _arg; // common argument
  // double common_radial_part;
  // double common_argument;

  double r; // WHAT IS THIS?
  double phi; // WHAT IS THIS? 
  double t; // WHAT IS THIS?

  // Get rotation angle to align with z-aligned Stokes basis
  void
  get_Stokes_alignment_angle(FourVector<double> &u, FourVector<double> &b, double &cs, double &sn);

  // Computes dlambda/dl
  double dl_dlambda(const double dydx[]);

  // double Ma_alpha(double alpha);
  // double alpha_term(double alpha);
  // double E_21(double alpha);
  // double beta_axion(double alpha);
  // double omega_21(double alpha);
  // double pre_factor_A(double alpha);
  double r_p(double M, double spin);
  double r_m(double M, double spin);
  double omega_crit(double M, double spin);
  double omega_21(double mu, double M, double spin);
  double sigma(double mu, double M, double spin);
  double q_term(double mu, double M, double spin);
  double x_term(double mu, double M, double spin);
  double Re_a_not_normed(double r);
  double find_max_Re_a_not_normed(double r_min, double r_max, double step);

  double dadr(double t, double r, double theta, double phi);
  double dadtheta(double t, double r, double theta, double phi);
  double dadphi(double t, double r, double theta, double phi);
  double dadt(double t, double r, double theta, double phi);
};


};
#endif

