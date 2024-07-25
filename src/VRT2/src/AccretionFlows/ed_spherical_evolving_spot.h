/***********************************************************************/
/*** Defines the spherical Gaussian spot density distribution        ***/
/*                                                                     */
/* Defines functions for:                                              */
/*                                                                     */
/*   (1) A linearly evolving hot-spot density.                         */
/*                                                                     */
/* Based upon the spherical outflowing spot class, and thus the spot   */
/* may follow an arbitrary trajectory as defined by the accretion      */
/* flow velocity.                                                      */
/*                                                                     */
/***********************************************************************/


#ifndef VRT2_ED_SPHERICAL_EVOLVING_SPOT_H
#define VRT2_ED_SPHERICAL_EVOLVING_SPOT_H

#include <vector>
#include <valarray>
#include <ostream>
#include <iomanip>

#include "metric.h"
#include "fourvector.h"
#include "ed_spherical_outflowing_spot.h"
#include "accretion_flow_velocity.h"

namespace VRT2 {
class ED_SphericalEvolvingSpot : public ED_SphericalOutflowingSpot
{
 public:
  ED_SphericalEvolvingSpot(Metric& g, double density_scale, double log_density_scale_dot, double rspot, AccretionFlowVelocity& afv, double t0, double r0, double theta0, double phi0, double tstart, double tend);
  virtual ~ED_SphericalEvolvingSpot() {};

  virtual double get_density(double t, double r, double theta, double phi);

  // Access to spot position
  std::valarray<double>& spot_center(double t);

 protected:
  double _log_density_scale0; // Initial density scale
  double _log_density_scale_dot; // Time derivative of the density scale (as seen from infinity)
};

inline double ED_SphericalEvolvingSpot::get_density(double t, double r, double theta, double phi)
{
  _density_scale = std::pow(10.0, _log_density_scale0 + _log_density_scale_dot*(t+_toffset));

  return ED_SphericalOutflowingSpot::get_density(t,r,theta,phi);
}


};
#endif
