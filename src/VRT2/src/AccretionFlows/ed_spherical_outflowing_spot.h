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

#ifndef VRT2_ED_SPHERICAL_OUTFLOWING_SPOT_H
#define VRT2_ED_SPHERICAL_OUTFLOWING_SPOT_H

#include <vector>
#include <valarray>
#include <ostream>
#include <iomanip>

#include "metric.h"
#include "fourvector.h"
#include "electron_density.h"
#include "accretion_flow_velocity.h"
#include "vrt2_globs.h"
#include "vrt2_constants.h"

namespace VRT2 {
class ED_SphericalOutflowingSpot : public ElectronDensity
{
 public:
  ED_SphericalOutflowingSpot(Metric& g, double density_scale, double rspot, AccretionFlowVelocity& afv, double t0, double r0, double theta0, double phi0, double tstart, double tend);
  virtual ~ED_SphericalOutflowingSpot() {};

  void reset(double t0, double r0, double theta0, double phi0, double tstart, double tend);

  void output_spot_path(std::ostream& os);

  virtual double get_density(double t, double r, double theta, double phi);

  void set_toffset(double toffset) { _toffset = toffset; }; // Allows us to step through time

  // Access to spot position
  std::valarray<double>& spot_center(double t);

 protected:
  Metric& _g;
  double _density_scale; // central density of spot
  double _rspot;  // Size of spot (in Gaussian sense)
  AccretionFlowVelocity& _afv; // Outflow velocity field
  double _t0, _r0, _theta0, _phi0; // Launch coordinates of the spot
  double _toffset; // Time offset of time origin

  // distance from spot center
  inline double delta_r(double t, double r, double theta, double phi, std::valarray<double> xspot_center);//, FourVector<double>& xspot_center);

  // interpolation and return value for spot center
  std::valarray<double> _xspot_center;

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

inline double ED_SphericalOutflowingSpot::get_density(double t, double r, double theta, double phi)
{
  // (A) Test to make sure that time has started
  if (t>=_xspot_tables[0][0]) {
    // (B) Get differential radius from spot center
    _xspot_center = spot_center(t);    
    double dr = delta_r(t,r,theta,phi,_xspot_center);

    /*
    std::cerr << "Spot center:"
	      << std::setw(15) << _xspot_center[0]
	      << std::setw(15) << _xspot_center[1]
	      << std::setw(15) << _xspot_center[2]
	      << std::setw(15) << _xspot_center[3]
	      << std::setw(15) << dr
	      << std::setw(15) << _density_scale
	      << std::setw(15) << _rspot
	      << '\n';
    */

    // (C) Get density
    return _density_scale * std::exp(-dr*dr/(2.0*_rspot*_rspot));
  }
  return 0.0;
}

inline double ED_SphericalOutflowingSpot::delta_r(double t, double r, double theta, double phi, std::valarray<double> xspot_center)//,FourVector<double>& xspot_center)
{
  //   (1) Reset metric to that spot
  _g.reset(xspot_center);
  
  //   (2) Create Cartesianized differences
  //double dt,dx,dy,dz;
  double dx,dy,dz;
  double rc = xspot_center[1];
  double st = std::sin(xspot_center[2]);
  double ct = std::cos(xspot_center[2]);
  double sp = std::sin(xspot_center[3]);
  double cp = std::cos(xspot_center[3]);
  //dt = t - xspot_center[0];
  dx = r*std::sin(theta)*std::cos(phi) - rc*st*cp;
  dy = r*std::sin(theta)*std::sin(phi) - rc*st*sp;
  dz = r*std::cos(theta) - rc*ct;
  
  //   (3) Return to spherical coords at spot position
  FourVector<double> x(_g);
  x.mkcon(0.0, st*cp*dx+st*sp*dy+ct*dz, (ct*cp*dx + ct*sp*dy - st*dz)/rc, (sp*dx-cp*dy)/(rc*st));
  
  //   (3b) Get flow velocity
  FourVector<double> u(_g);
  u = _afv(xspot_center[0],xspot_center[1],xspot_center[2],xspot_center[3]);
  
  //   (4) Get differential radius;
  double r2 = ( (x*x) + std::pow( (x*u), 2) );

  //   (5) Reset metric to ray position
  _g.reset(t,r,theta,phi);

  return std::sqrt(r2);
}

inline std::valarray<double>& ED_SphericalOutflowingSpot::spot_center(double t)
{
  size_t i = _xspot_tables[0].size()-1;
  double dt;
  _xspot_center[0] = t;
  if (t+_toffset>_xspot_tables[0][0] && t+_toffset<_xspot_tables[0][i]) {
    std::vector<double>::const_iterator p = std::lower_bound(_xspot_tables[0].begin(),_xspot_tables[0].end(),t+_toffset);
    // p should now be an iterator to the first value less than x (special cases should already be seperated out!)
    i = p - _xspot_tables[0].begin() - 1;
    dt = (t+_toffset-_xspot_tables[0][i])/(_xspot_tables[0][i+1]-_xspot_tables[0][i]);
    for (int j=1; j<4; ++j)
      _xspot_center[j] = dt*_xspot_tables[j][i+1] + (1.0-dt)*_xspot_tables[j][i];
  }
  else if (t+_toffset<=_xspot_tables[0][0])
    for (int j=1; j<4; ++j)
      _xspot_center[j] = _xspot_tables[j][0];
  else
    for (int j=1; j<4; ++j)
      _xspot_center[j] = _xspot_tables[j][i];

  return _xspot_center;  
}

/*** Relative error scalings ***/
#define TINY 1.0e-2 // VERY Sensitive -> Controls time!
void ED_SphericalOutflowingSpot::get_yscal(double h, double x, const double y[], const double dydx[], double yscal[]) const
{
  yscal[0] = std::fabs(y[0])+1.0*std::fabs(dydx[0]*h)+TINY; // t
  yscal[1] = std::fabs(y[1])+1.0*std::fabs(dydx[1]*h)+TINY; // r
  yscal[2] = VRT2_Constants::pi; // theta
  yscal[3] = VRT2_Constants::pi; // phi
}
#undef TINY


};
#endif
