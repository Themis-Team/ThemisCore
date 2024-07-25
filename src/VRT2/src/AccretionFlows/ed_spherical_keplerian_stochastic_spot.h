/*********************************************************/
/*** Generates Gaussian density as measured from a     ***/
/* point orbiting in the equatorial plane at the         */
/* Keplerian velocity, with stochastically (log-normal)  */
/* varying central density on some time Dt.              */
/*                                                       */
/*                                                       */
/* Note: ONLY IMPLEMENTED FOR KERR IN BOYER-LINDQUIST    */
/*                                                       */
/*********************************************************/

#ifndef VRT2_ED_SPHERICAL_KEPLERIAN_STOCHASTIC_SPOT_H
#define VRT2_ED_SPHERICAL_KEPLERIAN_STOCHASTIC_SPOT_H

#include "metric.h"
#include "fourvector.h"
#include "electron_density.h"
#include "random_number_generator.h"

namespace VRT2 {
class ED_SphericalKeplerianStochasticSpot : public ElectronDensity
{
 public:
  ED_SphericalKeplerianStochasticSpot(Metric& g, double dm, double sigld, double Dt, double t0, double rspot, double rorbit, double height, double phi0=0.0);
  virtual ~ED_SphericalKeplerianStochasticSpot();

  void set_phi0(double phi) {_phi0 = phi;};
  void set_density_scale(double density_scale) { _density_scale_factor=density_scale; };

  virtual double get_density(double t, double r, double theta, double phi);

 private:
  Metric& _g;
  double _ldm; // log of mean central density of spot
  double _sigld; // SD of log of mean central density
  double _Dt; // Time step between sub-flares (SHOULD ALSO BE RANDOMIZED!)
  double _t0; // Flare time shift
  double _rspot;  // Size of spot (in Gaussian sense)
  double _rhoorbit; // Equatorial radius of orbit
  double _zorbit; // Height of spot from equatorial plane
  double _phi0; // Initial phase of spot
  double _Omega;  // Splot angular velocity
  double _rorbit, _thetaorbit; // Spherical radius and theta of spot
  FourVector<double> _uspot; // local spot four-velocity
  FourVector<double> _xspot_center; // spot center position

  GaussianRandomNumberGenerator<Ran2RNG> _rng;
  double _tmin, _tmax;
  double* _density_scale; // central density of spot
  double _density_scale_factor;
  double _decay_rate;
};

inline double ED_SphericalKeplerianStochasticSpot::get_density(double t, double r, double theta, double phi)
{
  // (AA) Get the current spot density:
  double td = t+_phi0/_Omega + _t0;
  double dscale;
  if (td<_tmin || td>_tmax)
    dscale = 0.0;
  else
  {
    int id = int((td-_tmin)/_Dt);
    dscale = _density_scale_factor*_density_scale[id] * std::exp(-_decay_rate*(td-_Dt*id-_tmin));
  }


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



  /// DEBUG
  if (r2<2.0)
    std::cout << std::setw(15) << _phi0/_Omega
              << std::setw(15) << td
	      << std::setw(15) << dscale
	      << std::setw(15) << r
	      << std::setw(15) << r2
	      << std::endl;


  // (B) Get density
  return dscale * std::exp(-r2);
}
};
#endif
