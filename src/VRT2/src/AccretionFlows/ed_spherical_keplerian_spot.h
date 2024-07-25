/*********************************************************/
/*** Generates Gaussian density as measured from a     ***/
/* point orbiting in the equatorial plane at the         */
/* Keplerian velocity.                                   */
/*                                                       */
/* Note: ONLY IMPLEMENTED FOR KERR IN BOYER-LINDQUIST    */
/*                                                       */
/*********************************************************/

#ifndef VRT2_ED_SPHERICAL_KEPLERIAN_SPOT_H
#define VRT2_ED_SPHERICAL_KEPLERIAN_SPOT_H

#include "metric.h"
#include "fourvector.h"
#include "electron_density.h"

namespace VRT2 {
class ED_SphericalKeplerianSpot : public ElectronDensity
{
 public:
  ED_SphericalKeplerianSpot(Metric& g, double density_scale, double rspot, double rorbit, double height, double phi0=0.0);
  virtual ~ED_SphericalKeplerianSpot() {};

  void set_phi0(double phi) {_phi0 = phi;};
  void set_density_scale(double density_scale) { _density_scale=density_scale; };

  virtual double get_density(double t, double r, double theta, double phi);

 private:
  Metric& _g;
  double _density_scale; // central density of spot
  double _rspot;  // Size of spot (in Gaussian sense)
  double _rhoorbit; // Equatorial radius of orbit
  double _zorbit; // Height of spot from equatorial plane
  double _phi0; // Initial phase of spot
  double _Omega;  // Splot angular velocity
  double _rorbit, _thetaorbit; // Spherical radius and theta of spot
  FourVector<double> _uspot; // local spot four-velocity
  FourVector<double> _xspot_center; // spot center position
};

inline double ED_SphericalKeplerianSpot::get_density(double t, double r, double theta, double phi)
{
  // (A) Get differential radius from spot center
  //   (1) Set spot position
  _xspot_center.mkcon(t,_rorbit,_thetaorbit,_Omega*t+_phi0);
  //   (2) Reset metric to that spot
  _g.reset(_xspot_center.con());
  //   (3) Make local position fourvector
  FourVector<double> x(_g);
  x.mkcon(t,r,theta,phi);
  //   (3b) Make local velocity fourvector
  FourVector<double> u(_g);
  u.mkcon(1.0,0.0,0.0,_Omega);
  u *= 1.0/std::sqrt(-(u*u));
  //   (4) Get differential radius;
  x -= _xspot_center;
  std::valarray<double> xtmp = x.con();
  //     Put everything in right range (i.e. 0 and 2pi are close)
  //      (Note that theta can take on negative values, and thus must fold this correctly into phi)
  xtmp[2] = std::fabs(std::fmod(xtmp[2]+M_PI,2.0*M_PI)-M_PI);
  xtmp[3] = std::fmod(std::fmod(xtmp[3],2.0*M_PI)+3.0*M_PI,2.0*M_PI)-M_PI;
  x.mkcon(xtmp);
  double r2 = ((x*x)+std::pow( (x*u), 2))/(2.0*_rspot*_rspot);
  //   (5) Reset metric to ray position
  _g.reset(t,r,theta,phi);

  // (B) Get density
  return _density_scale * std::exp(-r2);
}
};
#endif
