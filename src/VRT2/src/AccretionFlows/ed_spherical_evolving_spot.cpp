#include "ed_spherical_evolving_spot.h"

namespace VRT2 {

ED_SphericalEvolvingSpot::ED_SphericalEvolvingSpot(Metric& g, double density_scale, double log_density_scale_dot, double rspot, AccretionFlowVelocity& afv, double t0, double r0, double theta0, double phi0, double tstart, double tend)
  : ED_SphericalOutflowingSpot(g,density_scale,rspot,afv,t0,r0,theta0,phi0,tstart,tend), _log_density_scale0(std::log10(density_scale)), _log_density_scale_dot(log_density_scale_dot)
{
}

};


