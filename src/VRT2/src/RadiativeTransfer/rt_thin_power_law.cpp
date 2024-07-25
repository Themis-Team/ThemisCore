#include "rt_thin_power_law.h"


namespace VRT2 {
RT_ThinPowerLaw::RT_ThinPowerLaw(Metric& g,
				 ElectronDensity& ne, AccretionFlowVelocity& u,
				 double spectral_index)
  : RadiativeTransfer(g), _ne(ne), _u(u), _spectral_index(spectral_index)
{
}
RT_ThinPowerLaw::RT_ThinPowerLaw(const double y[], Metric& g,
				 ElectronDensity& ne, AccretionFlowVelocity& u,
				 double spectral_index)
  : RadiativeTransfer(y,g), _ne(ne), _u(u), _spectral_index(spectral_index)
{
}
RT_ThinPowerLaw::RT_ThinPowerLaw(FourVector<double>& x, FourVector<double>& k, Metric& g,
				 ElectronDensity& ne, AccretionFlowVelocity& u,
				 double spectral_index)
  : RadiativeTransfer(x,k,g), _ne(ne), _u(u), _spectral_index(spectral_index)
{
}

std::valarray<double>& RT_ThinPowerLaw::IQUV_ems(const double dydx[])
{
  _iquv_ems = 0.0;
  _iquv_ems[0] = VRT2_Constants::me * VRT2_Constants::c * VRT2_Constants::c
    * _ne(_x) * std::pow( -(_u(_x)*_k),_spectral_index-3.0) * dl_dlambda(dydx);

  return _iquv_ems;
}

double RT_ThinPowerLaw::dl_dlambda(const double dydx[])
{
  // Note that dx_dlam^2 = 0 b.c. this is a null geodesic!
  FourVector<double> dx_dlam(_g);
  dx_dlam.mkcon(dydx);

  return std::fabs(dx_dlam*_u(_x));
}

double RT_ThinPowerLaw::stable_step_size(double h, const double y[], const double dydx[])
{
  return std::min(h,1.0e-1/dl_dlambda(dydx));
}
};
