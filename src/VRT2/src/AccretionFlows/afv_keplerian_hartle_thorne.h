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

#ifndef AFV_KEPLERIAN_HARTLE_THORNE
#define AFV_KEPLERIAN_HARTLE_THORNE

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
namespace VRT2{
class AFV_Keplerian_HartleThorne : public AccretionFlowVelocity
{
 public:
  AFV_Keplerian_HartleThorne(Metric& g, double ri);
  virtual ~AFV_Keplerian_HartleThorne();

  // User defined density
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);

 private:
  double _ri;
  double _Omegai;

  double Omega(double r);
  int _N;
  double *_lr, *_Omega;
  double _lrmin, _lrmax;

  FourVector<double>& get_keplerian_velocity(double r);
  FourVector<double>& get_freefall_velocity(double r, double theta);
};

inline FourVector<double>& AFV_Keplerian_HartleThorne::get_velocity(double, double r, double theta, double)
{
  if (r>=_ri)
    return get_keplerian_velocity(r);
  else
    return get_freefall_velocity(r,theta);
}

inline FourVector<double>& AFV_Keplerian_HartleThorne::get_keplerian_velocity(double r)
{
  double lr = std::log(r);
  int i = int( (lr-_lrmin)/(_lrmax-_lrmin) * _N );
  double Omega = 0;
  if (i>=0  && i<(_N-1))
  {
    double dlr = (lr-_lr[i])/(_lr[i+1]-_lr[i]);
    Omega = _Omega[i]*(1-dlr) + _Omega[i+1]*dlr;
  }

  _u.mkcon(1.0, 0.0, 0.0, Omega);   // Define u so it goes forward in time
  _u *= 1.0/std::sqrt( -(_u*_u) );

  return _u;
}

inline FourVector<double>& AFV_Keplerian_HartleThorne::get_freefall_velocity(double r, double theta)
{
  double e=1,l=0;
  if (r>_g.horizon()) 
  {
    // Get energy and angular momentum for point at same theta at _ri
    std::valarray<double> x=_g.local_position();
    _g.reset(0.0,_ri,theta,0.0);
    FourVector<double> uri = get_keplerian_velocity(_ri);
    e = uri.cov(0);
    l = uri.cov(3);    
    _g.reset(x);


    // Covariant radial velocity (make infalling
    double vr = -std::sqrt( std::max(0.0,-(1.0+_g.ginv(0,0)*e*e+2.0*_g.ginv(0,3)*e*l+_g.ginv(3,3)*l*l)/_g.ginv(1,1)) );
    _u.mkcov(e, vr, 0.0, l);
    

    /*
    // Covariant radial velocity (make infalling
    double vr = -std::sqrt( std::max(0.0,-(1.0+_g.ginv(0,0)*e*e)/_g.ginv(1,1)) );
    _u.mkcov(e, vr, 0.0, 0.0);
    */

    //_u.mkcov(e, 0.0, 0.0, l);
    //_u *= 1.0/std::sqrt( -(_u*_u));

  }
  else
    _u.mkcov(-1.0, 0.0, 0.0, 0.0);

  if ((_u*_u)>=0)
    std::cout << "AFV_Keplerian_HartleThorne, u has become spacelike"
	      << std::setw(15) << r
	      << std::setw(15) << theta
	      << std::setw(15) << e
	      << std::setw(15) << l
	      << std::setw(15) << _u.con(0)
	      << std::setw(15) << _u.con(1)
	      << std::setw(15) << _u.con(2)
	      << std::setw(15) << _u.con(3)
	      << std::setw(15) << (_u*_u)
	      << std::endl;

  return _u;
}

};
#endif
