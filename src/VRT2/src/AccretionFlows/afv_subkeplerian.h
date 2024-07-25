/***************************************************/
/*** Defines the velocity for a Keplerian flow   ***/
/*                                                 */
/* The velocity is given by the (gr) keplerian     */
/* value for equitorial radii outside of ri, and a */
/* constant angular momentum value for radii       */
/* inside ri.  The angular momentum in the         */
/* interior is chosen such that the velocity is    */
/* continuous.                                     */
/*                                                 */
/* CURRENTLY ONLY WORKS FOR BOYER-LINDQUIST COORDS */
/*                                                 */
/***************************************************/

#ifndef VRT2_AFV_SUBKEPLERIAN_H
#define VRT2_AFV_SUBKEPLERIAN_H

#include <cmath>
#include <math.h>
#include <valarray>
#include <algorithm>

#include "accretion_flow_velocity.h"
#include "metric.h"
#include "kerr.h"
#include "fourvector.h"

//#define INNER_DISK_CONST_E     // uses constant energy condition in inner disk if defined
                               // otherwise uses constant angular momentum
namespace VRT2 {
class AFV_SubKeplerian : public AccretionFlowVelocity
{
 public:
  AFV_SubKeplerian(Metric& g, double ri, double subkeplerian_factor=1.0);
  virtual ~AFV_SubKeplerian() { delete _g_local; };

  // User defined density
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);

 private:
  double _ri;
  //double _Omegai;
  Metric *_g_local;
  double _subkeplerian_factor;

  FourVector<double>& get_subkeplerian_velocity(double r);
  FourVector<double>& get_freefall_velocity(double r, double theta);
};


inline FourVector<double>& AFV_SubKeplerian::get_velocity(double, double r, double theta, double)
{
  if (r>=_ri)
    return get_subkeplerian_velocity(r);
  else
    return get_freefall_velocity(r,theta);
}

inline FourVector<double>& AFV_SubKeplerian::get_subkeplerian_velocity(double r)
{
  double Omega_Keplerian = 1.0/( std::pow(r/_g.mass(),1.5) + _g.ang_mom() );
  

  _u.mkcon(1.0, 0.0, 0.0, Omega_Keplerian);   // Define u so it goes forward in time
  double L = -_u.cov(3)/_u.cov(0) * _subkeplerian_factor;
  _u.mkcov(-1.0, 0.0, 0.0, L);
  _u *= 1.0/std::sqrt( -(_u*_u) );

  return _u;
}

inline FourVector<double>& AFV_SubKeplerian::get_freefall_velocity(double r, double theta)
{
  if (r>_g.horizon()) {
    // Get energy and angular momentum for point at same theta at _ri
    _g_local->reset(0.0,_ri,theta,0.0);
    double Omega = 1.0/( std::pow(_ri/_g.mass(),1.5) + _g.ang_mom() );
    FourVector<double> uri(*_g_local);
    uri.mkcon(1.0, 0.0, 0.0, Omega);
    double L = -uri.cov(3)/uri.cov(0) * _subkeplerian_factor;
    uri.mkcov(-1.0,0,0,L);
    uri *= 1.0/std::sqrt( -(uri*uri) );

    double e = uri.cov(0);
    double l = uri.cov(3);

    // Covariant radial velocity (make infalling
    double vr = -std::sqrt( std::max(0.0,-(1.0+_g.ginv(0,0)*e*e+2.0*_g.ginv(0,3)*e*l+_g.ginv(3,3)*l*l)/_g.ginv(1,1)) );
    _u.mkcov(e, vr, 0.0, l);
    //_u.mkcov(e, 0.0, 0.0, l);
    //_u *= 1.0/std::sqrt( -(_u*_u));
  }
  else
    _u.mkcov(-1.0, 0.0, 0.0, 0.0);

  return _u;
}
};
#endif
