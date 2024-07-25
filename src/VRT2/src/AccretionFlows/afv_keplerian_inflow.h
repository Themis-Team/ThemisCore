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

#ifndef VRT2_AFV_KEPLERIANINFLOW
#define VRT2_AFV_KEPLERIANINFLOW

#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>
#include <algorithm>

#include "accretion_flow_velocity.h"
#include "metric.h"
#include "kerr.h"
#include "fourvector.h"

//#define INNER_DISK_CONST_E     // uses constant energy condition in inner disk if defined
                               // otherwise uses constant angular momentum

namespace VRT2 {
class AFV_KeplerianInflow : public AccretionFlowVelocity
{
 public:
  AFV_KeplerianInflow(Metric& g, double ri, double Pdot);
  virtual ~AFV_KeplerianInflow() {};

  // User defined density
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);

 private:
  double _ri;  // Inner radius of Keplerian flow
  double _Pdot; // Time rate of change of period -> translates into radial velocity by Pdot = 3 pi vr/vk
  double _Omegai;  // Angular frequency at inner radius of Keplerian flow
  Metric *_g_local;

  FourVector<double>& get_KeplerianInflow_velocity(double r);
  FourVector<double>& get_freefall_velocity(double r, double theta);
};

inline FourVector<double>& AFV_KeplerianInflow::get_velocity(double, double r, double theta, double)
{
  if (r > _ri)
    return get_KeplerianInflow_velocity(r);
  else
    return get_freefall_velocity(r,theta);
}

inline FourVector<double>& AFV_KeplerianInflow::get_KeplerianInflow_velocity(double r)
{
  double Omega = std::sqrt(_g.mass())/( std::pow(r,1.5) + (_g.ang_mom()/std::sqrt(_g.mass() )) );

  _u.mkcon(1.0, _Pdot*r*Omega/(3.0*M_PI), 0.0, Omega);   // Define u so it goes forward in time
  _u *= 1.0/std::sqrt( -(_u*_u) );

  return _u;
}

inline FourVector<double>& AFV_KeplerianInflow::get_freefall_velocity(double r, double theta)
{
  if (r>_g.horizon()) {
    // Get energy and angular momentum for point at same theta at _ri
    _g_local->reset(0.0,_ri,theta,0.0);
    double Omega = std::sqrt(_g.mass())/( std::pow(_ri,1.5) + (_g.ang_mom()/std::sqrt(_g.mass())) );


     FourVector<double> uri(*_g_local);
    uri.mkcon(1.0, _Pdot*_ri*Omega/(3.0*M_PI), 0.0, Omega);
    uri *= 1.0/std::sqrt( -(uri*uri) );
    double e = uri.cov(0);
    double l = uri.cov(3);

    // Covariant radial velocity (make infalling
    double ur = -std::sqrt(std::max(0.0, -(1.0+_g.ginv(0,0)*e*e+2.0*_g.ginv(0,3)*e*l+_g.ginv(3,3)*l*l)/_g.ginv(1,1)));

    if ( ur == 0)
    {
      std::cerr << "ur went to zero\n";
      std::cerr << std::setw(15) << r
                << std::setw(15) << theta
                << std::setw(15) << ur << std::endl;
      std::abort();
    }


    _u.mkcov(e, ur, 0.0, l);
    //_u.mkcov(e, 0.0, 0.0, l);
  }
  else
    _u.mkcov(-1.0, 0.0, 0.0, 0.0);

  /*
  if ( (_u*_u)>0.0 )
  {
    std::cerr << "AFV_KeplerianInflow: _u has become spacelike\n";
    std::abort();
  }
  */

  return _u;
}
};
#endif
