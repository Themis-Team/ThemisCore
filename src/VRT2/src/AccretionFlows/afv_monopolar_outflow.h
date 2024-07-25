/***********************************************************************/
/*** Defines the velocity field for a monopolar force-free/MHD field ***/
/*                                                                     */
/* Keeps u_t and u_phi constant and u^theta=0.  This defines u^r!      */
/* Gets u_phi as a function of theta by fixing omega on a sphere at    */
/* radius r0.                                                          */
/*                                                                     */
/* ASSUMES THAT g HAS ALREADY BEEN RESET                               */
/*                                                                     */
/* CURRENTLY ONLY WORKS FOR BOYER-LINDQUIST COORDS                     */
/*                                                                     */
/***********************************************************************/

#ifndef VRT2_AFV_MONOPOLAR_OUTFLOW_H
#define VRT2_AFV_MONOPOLAR_OUTFLOW_H

#include <valarray>
#include <cmath>
#include <math.h>
using namespace std;
#include <algorithm>

#include "accretion_flow_velocity.h"
#include "metric.h"
#include "kerr.h"
#include "fourvector.h"

#include "vrt2_constants.h"

namespace VRT2 {

//#define INNER_DISK_CONST_E     // uses constant energy condition in inner disk if defined
                               // otherwise uses constant angular momentum

class AFV_MonopolarOutflow : public AccretionFlowVelocity
{
 public:
  AFV_MonopolarOutflow(Metric& g, double e, double r0, double Omega);
  virtual ~AFV_MonopolarOutflow() {};

  // User defined density
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);

 private:
  double _e;  // Energy at infinity (-u_t)
  double _r0; // Launch radius
  double _Omega; // Fraction of maximum angular momentum
  int _sgn;
  Metric *_g_local;


  double specific_angmom(double Omega, double r, double theta);
  double max_specific_angmom(int sgn, double r, double theta);
};

#define TINY 1.0e-10
inline FourVector<double>& AFV_MonopolarOutflow::get_velocity(double t, double r, double theta, double phi)
{
  // Rotate theta into [0,pi] (the dumb way)
  double x = std::sin(theta)*std::cos(phi);
  double y = std::sin(theta)*std::sin(phi);
  double z = std::cos(theta);

  theta = atan2(std::sqrt(x*x+y*y),z);
  phi = atan2(y,x);

  if (theta<TINY || VRT2_Constants::pi-theta<TINY) {
    _g.reset(t,r,TINY,phi);
  }

  double gtt = _g.ginv(0,0);
  double gtp = _g.ginv(0,3);
  double gpp = _g.ginv(3,3);
  double grr = _g.g(1,1);

  double l = specific_angmom(_Omega,_r0,theta)*_e;

  double ur = std::sqrt( std::max(0.0,-(1.0 + gtt*_e*_e - 2.0*gtp*_e*l + gpp*l*l)/grr) );
  double ut = -gtt*_e + gtp*l;
  double up = gpp*l - gtp*_e;

  if (theta<TINY || VRT2_Constants::pi-theta<TINY) {
    _g.reset(t,r,theta,phi);
  }

  _u.mkcon(ut,ur,0.0,up);

  return _u;
}

inline double AFV_MonopolarOutflow::specific_angmom(double Omega, double r, double theta)
{
  if (theta<TINY)
    theta = TINY;
  _g_local->reset(0,r,theta,0);


  FourVector<double> utmp(*_g_local);
  utmp.mkcon(1.0,0.0,0.0,Omega);
  utmp *= 1.0/std::sqrt( -(utmp*utmp) );
  return utmp.cov(3);
  
  /*
  double gtp_local = _g_local->g(0,3);
  return ( -  ( _g_local->g(3,3)*_Omega + gtp_local )/( _g_local->g(0,0) + gtp_local*_Omega ) );
  */
}

inline double AFV_MonopolarOutflow::max_specific_angmom(int sgn, double r, double theta)
{
  if (theta<TINY)
    theta = TINY;
  _g_local->reset(0,r,theta,0);
  double gtt = _g_local->ginv(0,0);
  double gtp = _g_local->ginv(0,3);
  double gpp = _g_local->ginv(3,3);
  return ( gtp/gpp + sgn*std::sqrt( (gtp*gtp)/(gpp*gpp) - (gtt+1.0/(_e*_e))/gpp ) );
}
#undef TINY
};
#endif
