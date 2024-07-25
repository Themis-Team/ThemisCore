/***************************************************/
/*** Defines the velocity for a Infalling flow   ***/
/*                                                 */
/* The velocity is given by the  keplerian          */
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

#ifndef VRT2_AFV_INFLOWJOHANNSEN
#define VRT2_AFV_INFLOWJOHANNSEN

#include <cmath>
#include <math.h>

#include <valarray>
#include <algorithm>

#include "accretion_flow_velocity.h"
#include "johannsen.h"
#include "fourvector.h"


namespace VRT2 {
class AFV_InflowJohannsen : public AccretionFlowVelocity
{
 public:
  AFV_InflowJohannsen(Johannsen& g, double ri, double infallRate, double subKep);
  virtual ~AFV_InflowJohannsen() {};

  // User defined density
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);

 private:

  FourVector<double>& uKep(double, double);
  double _ri;  // Inner radius of Keplerian flow
  double _infallRate;
  double _subKep;
  FourVector<double> _uK;

  Metric *_g_local;

};

inline FourVector<double>& AFV_InflowJohannsen::get_velocity(double t, double r, double theta, double phi)
{
  if ( r > _g.horizon()){
    //Create free fall velocity
    double urff = -std::sqrt(std::max(0.0, -(1.0+_g.ginv(0,0))*_g.ginv(1,1)));
    double Omegaff = _g.ginv(0,3)/_g.ginv(0,0);

    //Get keplerian component
    _uK = uKep(r, theta);
    double urK = _uK.con(1);
    double OmegaK = _uK.con(3)/_uK.con(0);
 
    //Conbined vector field
    double ur = urK + _infallRate*(urff - urK);
    double Omega = Omegaff + _subKep*(OmegaK - Omegaff);
    double K = _g.g(0,0) + 2*Omega*_g.g(0,3) + Omega*Omega*_g.g(3,3);
    double ut = std::sqrt(-(1+ur*ur*_g.g(1,1))/K);
    if ( K >=0.0 )
    {
      std::cerr << "_u has gone spacelike ut = 0\n"
                << std::setw(15) << r
                << std::setw(15) << _infallRate
					  		<< std::setw(15) << _subKep << std::endl;
      std::abort();
    }

    _u.mkcon(ut,ur,0,Omega*ut);
  }
  else{
    _u.mkcon(-1.0,0.0,0.0,0.0);
  }
  return _u;
}

inline FourVector<double>& AFV_InflowJohannsen::uKep(double r, double theta)
{
  if (r>=_ri) {
    _g_local->reset(0.0,r,M_PI/2.0,0.0);
    double gttr = _g_local->Metric::Dg(0,0,1);
    double gtpr = _g_local->Metric::Dg(0,3,1);
    double gppr = _g_local->Metric::Dg(3,3,1);
 
    double OmegaK =  (-gtpr + std::sqrt(gtpr*gtpr - gttr*gppr))/gppr;
    _uK.mkcon(1,0,0,OmegaK);
    _uK *= 1.0/std::sqrt(-(_uK*_uK));

    return _uK;
  }
     
  else{
    _g_local->reset(0.0, _ri, M_PI/2.0, 0.0);
    double gttr = _g_local->Metric::Dg(0,0,1);
    double gtpr = _g_local->Metric::Dg(0,3,1);
    double gppr = _g_local->Metric::Dg(3,3,1);
    double OmegaKi = (-gtpr + std::sqrt(gtpr*gtpr - gttr*gppr))/gppr;
    
    //Grab keplerian frequency in the plane
    _g_local->reset(0.0,_ri, theta, 0.0); 
    FourVector<double> uri(*_g_local);
    uri.mkcon(1.0, 0.0, 0.0, OmegaKi);
    uri *= 1.0/std::sqrt( -(uri*uri) );
    double e = uri.cov(0);
    double l = uri.cov(3);

    // Covariant radial velocity (make infalling
    double urk = -std::sqrt(std::max(0.0,-(1.0+_g.ginv(0,0)*e*e+2.0*_g.ginv(0,3)*e*l+_g.ginv(3,3)*l*l)/_g.ginv(1,1)));

    _uK.mkcov(e, urk, 0.0, l);
    return _uK;

  }
}
};
#endif
