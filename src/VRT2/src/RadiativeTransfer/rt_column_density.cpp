#include "rt_column_density.h"

namespace VRT2 {
RT_ColumnDensity::RT_ColumnDensity(Metric& g,
					       ElectronDensity& ne, AccretionFlowVelocity& u)
  : RadiativeTransfer(g), _ne(ne), _u(u)
{
}
RT_ColumnDensity::RT_ColumnDensity(const double y[], Metric& g,
					       ElectronDensity& ne, AccretionFlowVelocity& u)
  : RadiativeTransfer(y,g), _ne(ne), _u(u)
{
}
RT_ColumnDensity::RT_ColumnDensity(FourVector<double>& x, FourVector<double>& k, Metric& g,
					       ElectronDensity& ne, AccretionFlowVelocity& u)
  : RadiativeTransfer(x,k,g), _ne(ne), _u(u)
{
}

std::valarray<double>& RT_ColumnDensity::IQUV_ems(const double dydx[])
{
  _iquv_ems = 0.0;
  _iquv_ems[0] = _ne(_x) * dl_dlambda(dydx);

  return _iquv_ems;
}

double RT_ColumnDensity::dl_dlambda(const double dydx[])
{
  // Note that dx_dlam^2 = 0 b.c. this is a null geodesic!
  FourVector<double> dx_dlam(_g);
  dx_dlam.mkcon(dydx);

  return std::fabs(dx_dlam*_u(_x));
}

double RT_ColumnDensity::stable_step_size(double h, const double y[], const double dydx[])
{
  return std::min(h,1.0/dl_dlambda(dydx));
}
};
