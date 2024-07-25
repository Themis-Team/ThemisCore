/***********************************************************************/
/*** Defines the velocity field for Falcke model jet                 ***/
/*                                                                     */
/* Presently it keeps u_phi constant, and sets it at the jet base via  */
/* constant u^\phi = 0.5 Omega_H.                                      */
/*                                                                     */
/*  ONLY IMPLEMENTED FOR KERR!!!!                                      */
/*                                                                     */
/*                                                                     */
/*                                                                     */
/***********************************************************************/

#ifndef VRT2_AFV_FALCKE_H
#define VRT2_AFV_FALCKE_H

#include <valarray>
#include <cmath>
#include <math.h>
using namespace std;
#include <algorithm>

#include "accretion_flow_velocity.h"
#include "metric.h"
#include "kerr.h"
#include "fourvector.h"
#include "vrt2_globs.h"
#include "vrt2_constants.h"

#include "Falcke_jet_model.h"

namespace VRT2 {

class AFV_Falcke : public AccretionFlowVelocity
{
 public:
  AFV_Falcke(Metric& g, FalckeJetModel& jet);
  virtual ~AFV_Falcke() {};

  // User defined density
  virtual FourVector<double>& get_velocity(double t, double r, double theta, double phi);

 private:

  //Metric *_g_local;
  FalckeJetModel& _jet;
  double _rmin;
};

#define TINY 1.0e-10
inline FourVector<double>& AFV_Falcke::get_velocity(double t, double r, double theta, double phi)
{
  // Rotate theta into [0,pi] (the dumb way)
  double x = r*std::sin(theta)*std::cos(phi);
  double y = r*std::sin(theta)*std::sin(phi);
  double z = r*std::cos(theta);
  double rho = std::sqrt(x*x+y*y);

  theta = atan2(std::sqrt(x*x+y*y),z);
  phi = atan2(y,x);

  // Get uz and ux in Cartesianized frame
  double uz = std::fabs(_jet.gbj(z));
  double ux = std::fabs(_jet.gbs());

  // Convert into ur and uth
  double ur = uz*std::fabs(std::cos(theta)) + ux*std::fabs(std::sin(theta));
  double uth = (z<0 ? -1 : 1)*(ux*std::fabs(std::cos(theta)) - uz*std::fabs(std::sin(theta)))/r;

  // Get some metric components
  double gtt = _g.ginv(0,0);
  double gtp = _g.ginv(0,3);
  double gpp = _g.ginv(3,3);
  double ghh = _g.g(2,2);
  double grr = _g.g(1,1);


  // Get the angular velocity of the foot print:
  //   rho>rmb -> Keplerian
  //   rho<rmb -> Uniform such that continuous

  double rho_nozzle = rho * _jet.rperp(_jet.nozzle_height())/_jet.rperp(z);
  double r_tmp = std::sqrt( rho_nozzle*rho_nozzle + _jet.nozzle_height()*_jet.nozzle_height() );
  double theta_tmp = std::max( atan2(rho_nozzle,_jet.nozzle_height()), TINY );
  _g.reset(t,r_tmp,theta_tmp,phi);
  //    Now choose Omega to be given by either Keplerian or constant
  double Omega = 1.0/(std::pow(std::max(rho_nozzle,_rmin),1.5)+_g.ang_mom());
  //    Now convert to L
  double L = - (_g.g(3,3)*Omega + _g.g(0,3))/(_g.g(0,0) + _g.g(0,3)*Omega);
  //    Now reset metric to present position
  if (theta<TINY || VRT2_Constants::pi-theta<TINY) {
    _g.reset(t,r,TINY,phi);
  }
  //    Now set u_t and u_phi;
  double ult = -std::sqrt(  -  (1.0 + grr*ur*ur + ghh*uth*uth) / (gtt - 2.0*gtp*L + gpp*L*L) );
  if (vrt2_isnan(ult))
    std::cerr << "ult is nanified (bad L dipshit!)\n\n";
  double ulph = -L*ult;
  double ulr = grr*ur;
  double ulth = ghh*uth;

  // Reset the metric
  _g.reset(t,r,theta,phi);

  // Define _u!
  _u.mkcov(ult,ulr,ulth,ulph);

  /*
  std::cout << "u^2 = " << std::setw(15) << (_u*_u)
	    << "  ulph = " << std::setw(15) << ulph
	    << "  ur = " << std::setw(15) << ur
	    << "  uth = " << std::setw(15) << uth
	    << "  ut = " << std::setw(15) << ult
	    << "  gamma = " << std::sqrt(-_u.cov(0)*_u.con(0))
	    << "\n\n";
  */

  return _u;
}
#undef TINY
};

#endif
