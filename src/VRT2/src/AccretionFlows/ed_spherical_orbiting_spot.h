/***********************************************************************/
/*** Defines the spherical Gaussian spot density distribution        ***/
/*                                                                     */
/* Defines functions for:                                              */
/*                                                                     */
/*   (1) Integrating a curve through an arbitrary velocity field to    */
/*       get the spot center.  This begins at time t0 and position     */
/*       r0,theta0,phi0.                                               */
/*                                                                     */
/*   (2) Determining the distance in a coordinate-singularity-safe way */
/*       from the spot center.  The distance is a Cartesianized        */
/*       differential distance (First the coords are converted to      */
/*       Cartesian coords, the difference found, and then converted    */
/*       back to get line-element difference.  This takes care of      */
/*       problems near the azimuthal axis.)                            */
/*                                                                     */
/* Provides a constant size Gaussian spot along this curve.  Since the */
/* functions mentioned above are protected, they are available to      */
/* classes based upon this class.  Thus it is possible to produce      */
/* electron densities with arbitrary evolution along the trajectory.   */
/*                                                                     */
/***********************************************************************/

// FIX THE CONSTRUCTOR SO THAT YOU PASS THE LIMITS OF THE INTEGRATION
// SEPERATELY FROM (t0, r0, theta0, phi0) BY USING push_front TO INTEGRATE
// BACKWARDS FROM (t0, r0, theta0, phi0) TO TSTART AND push_back to
// INTEGRATE FROM (t0, r0, theta0, phi0) TO TEND.

#ifndef VRT2_ED_SPHERICAL_ORBITING_SPOT_H
#define VRT2_ED_SPHERICAL_ORBITING_SPOT_H

#include <vector>
#include <valarray>
#include <ostream>
#include <fstream>
#include <iomanip>

#include "metric.h"
#include "fourvector.h"
#include "electron_density.h"
#include "afv_shearing_inflow.h"
#include "vrt2_globs.h"
#include "vrt2_constants.h"

#include "fast_math.h"

namespace VRT2 {
class ED_SphericalOrbitingSpot : public ElectronDensity
{
 public:
  ED_SphericalOrbitingSpot(Metric& g, double density_scale, double rspot, AFV_ShearingInflow& afv, double t0, double r0, double theta0, double phi0, double tstart, double tobs);
  virtual ~ED_SphericalOrbitingSpot() {};

  void reset(double t0, double r0, double theta0, double phi0, double tstart, double tmax);

  void output_spot_path(std::string fname);

  virtual double get_density(double t, double r, double theta, double phi);

  void set_toffset(double toffset) { _toffset = toffset; }; // Allows us to step through time

  // Access to spot position
  std::valarray<double>& spot_center(double t);

 protected:
  Metric& _g;
  double _density_scale; // central density of spot
  double _rspot;  // Size of spot (in Gaussian sense)
  AFV_ShearingInflow& _afv; // Outflow velocity field
  double _t0, _r0, _theta0, _phi0; // Launch coordinates of the spot
  double _toffset; // Time offset of time origin

  // distance from spot center
  double delta_r(double t, double r, double theta, double phi, std::valarray<double> xspot_center);//, FourVector<double>& xspot_center);

  // interpolation and return value for spot center
  std::valarray<double> _xspot_center;

  double reparameterize(const double y[]);

  // spot center location tables
  std::vector<double> _xspot_tables[4];
  void generate_spot_position_table(double t, double r, double theta, double phi, double tstart, double tend);
  // Functions for derivs
  void derivs(double,const double[],double[]);
  inline void get_yscal(double,double,const double[],const double[],double[]) const;
  // Numerics
  int rkqs(double[],double[],int,double&,double,double,double[],double&,double&);
  void rkck(double [],double [],int,double,double,double [],double []);  

};

/*** Relative error scalings ***/
#define TINY 1.0e-2 // VERY Sensitive -> Controls time!
void ED_SphericalOrbitingSpot::get_yscal(double h, double x, const double y[], const double dydx[], double yscal[]) const
{
  yscal[0] = std::fabs(y[0])+1.0*std::fabs(dydx[0]*h)+TINY; // t
  yscal[1] = std::fabs(y[1])+1.0*std::fabs(dydx[1]*h)+TINY; // r
  yscal[2] = VRT2_Constants::pi; // theta
  yscal[3] = VRT2_Constants::pi; // phi
}
#undef TINY


};
#endif
