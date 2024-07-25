/***************************************************/
/*** Defines the velocity for a Infalling flow   ***/
/*                                                 */
/* The velocity is given by the (gr) keplerian     */
/* plus the free fall velocity for a spot          */
/* value for equitorial radii outside of ri, and a */
/* constant angular momentum value for radii       */
/* inside ri plus free fall.                       */
/* The angular momentum in the                     */
/* interior is chosen such that the velocity is    */
/* continuous.                                     */
/*                                                 */
/* CURRENTLY ONLY WORKS FOR BOYER-LINDQUIST COORDS */
/*                                                 */
/***************************************************/

#ifndef VRT2_AFV_SHEARINGINFLOW
#define VRT2_AFV_SHEARINGINFLOW

#include <cmath>
#include <math.h>
using namespace std;
#include <valarray>
#include <algorithm>

#include "accretion_flow_velocity.h"
#include "metric.h"
#include "kerr.h"
#include "fourvector.h"


namespace VRT2 {
class AFV_ShearingInflow : public AccretionFlowVelocity
{
 public:
  AFV_ShearingInflow(Metric& g, double ri, double infallRate, double subKep);
  virtual ~AFV_ShearingInflow() {};

  // User defined density
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);

 private:

  FourVector<double>& get_KeplerianInflow_velocity(double, double);
  FourVector<double>& get_freefall_velocity(double, double);
  double _ri;  // Inner radius of Keplerian flow
  double _infallRate;
  double _subKep;

  Metric *_g_local;

};

inline FourVector<double>& AFV_ShearingInflow::get_velocity(double, double r, double theta, double)
{
  if (r >= _ri)
    return get_KeplerianInflow_velocity(r, theta);
  else
    return get_freefall_velocity(r,theta);
}

inline FourVector<double>& AFV_ShearingInflow::get_KeplerianInflow_velocity(double r, double theta)
{
  //Keplerian component
  double OmegaK = std::sqrt(_g.mass())/( std::pow(r,1.5) + (std::fabs(_g.ang_mom())/std::sqrt(_g.mass() )) );

  FourVector<double> uK(_g);
  uK.mkcon(1.0, 0.0, 0.0, OmegaK);   // Define u so it goes forward in time
  uK *= 1.0/std::sqrt( -(uK*uK) );
  
  //Free-fall component
  double urff = -std::sqrt(std::max(0.0, -(1.0+_g.ginv(0,0))*_g.ginv(1,1)));
  double Omegaff = _g.ginv(0,3)/_g.ginv(0,0);
  
  //Conbined vector field
  double ur = _infallRate*urff;
  double Omega = OmegaK + (1-_subKep)*(Omegaff - OmegaK);
  double K = _g.g(0,0) + 2*Omega*_g.g(0,3) + Omega*Omega*_g.g(3,3);
  double ut = std::sqrt(std::max(0.0,-(1+ur*ur*_g.g(1,1))/K));
  if ( ut == 0)
  {
    std::cerr << "_u has gone spacelike ut = 0\n"
              << std::setw(15) << r
              << std::setw(15) << _infallRate
	      << std::setw(15) << _subKep << std::endl;
    std::abort();
  }

  _u.mkcon(ut,ur,0,Omega*ut);
  if ( _u*_u > -0.99)
    std::cerr << std::setw(15) << r 
              << std::setw(15) << _u*_u << std::endl;

  return _u;
}

inline FourVector<double>& AFV_ShearingInflow::get_freefall_velocity(double r, double theta)
{
  if (r>_g.horizon()) {

    //Keplerian component
    // Get energy and angular momentum for point at same theta at _ri
    
    _g_local->reset(0.0,_ri,theta,0.0);
    double OmegaKi = std::sqrt(_g.mass())/( std::pow(_ri,1.5) + (_g.ang_mom()/std::sqrt(_g.mass())) );


    FourVector<double> uri(*_g_local);
    uri.mkcon(1.0, 0.0, 0.0, OmegaKi);
    uri *= 1.0/std::sqrt( -(uri*uri) );
    double e = uri.cov(0);
    double l = uri.cov(3);

    // Covariant radial velocity (make infalling
    double urk = -std::sqrt(std::max(0.0,-(1.0+_g.ginv(0,0)*e*e+2.0*_g.ginv(0,3)*e*l+_g.ginv(3,3)*l*l)/_g.ginv(1,1)));


    FourVector<double> uk(_g);
    uk.mkcov(e, urk, 0.0, l);
    double OmegaK = uk.con(3)/uk.con(0);
    
    //Free fall stuff
    double urff = -std::sqrt(std::max(0.0,-(1.0+_g.ginv(0,0))*_g.ginv(1,1)));
    double Omegaff = _g.ginv(0,3)/_g.ginv(0,0);

    //Combined
    double Omega = OmegaK + (1-_subKep)*(Omegaff - OmegaK);
    double ur = uk.con(1) + _infallRate*(urff - uk.con(1));
    double K = _g.g(0,0) + 2*Omega*_g.g(0,3) + Omega*Omega*_g.g(3,3);
    double ut = std::sqrt(std::max(0.0,-(1+ur*ur*_g.g(1,1))/K));
    if ( ut == 0.0)
    {
      std::cerr << "u has gone spacelike\n"
                << std::setw(15) << r
                << std::setw(15) << _infallRate 
		<< std::setw(15) << _subKep << std::endl;
      std::abort();
    }

    _u.mkcon(ut, ur, 0.0, ut*Omega);

  }
  else
    _u.mkcov(-1.0, 0.0, 0.0, 0.0);


  return _u;
}
};
#endif
