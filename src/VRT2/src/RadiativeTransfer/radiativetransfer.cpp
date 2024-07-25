// Include Statements
#include "radiativetransfer.h"

namespace VRT2 {
RadiativeTransfer::RadiativeTransfer(Metric& g)
  : _g(g), _x(_g), _k(_g), _omega_scale(1.0), _length_scale(1.0)
{
  _iquv_abs.resize(5);
  _iquv_ems.resize(5);
}

RadiativeTransfer::RadiativeTransfer(const double y[], Metric& g)
  : _g(g), _x(_g), _k(_g), _omega_scale(1.0), _length_scale(1.0)
{
  _iquv_abs.resize(5);
  _iquv_ems.resize(5);
  reinitialize(y);
}

RadiativeTransfer::RadiativeTransfer(FourVector<double>& x, FourVector<double>& k, Metric& g)
  : _g(g), _x(_g), _k(_g), _omega_scale(1.0), _length_scale(1.0)
{
  _iquv_abs.resize(5);
  _iquv_ems.resize(5);
  reinitialize(x,k);
}

void RadiativeTransfer::reinitialize(const double y[])
{
  _x.mkcon(y);
  _k.mkcov(y+4);
}

void RadiativeTransfer::reinitialize(FourVector<double>& x, FourVector<double>& k)
{
  _x = x;
  _k = k;
}

// Characteristic RT distance
double RadiativeTransfer::dlambda(const double [], const double [])
{
  return 1.0;
}

// Absorption vectors
std::valarray<double>& RadiativeTransfer::IQUV_abs(const double [], const double [])
{
  _iquv_abs = 0.0;
  return (_iquv_abs);
}
double RadiativeTransfer::isotropic_absorptivity(const double [])
{
  return 0.0;
}
// Emissivity vector
std::valarray<double>& RadiativeTransfer::IQUV_ems(const double [])
{
  _iquv_ems = 0.0;
  return (_iquv_ems);
}

// Linear RT Change
void RadiativeTransfer::IQUV_rotate(double iquv[], double lambdai, const double yi[], const double dydxi[], double lambdaf, const double yf[], const double dydxf[])
{
}


// Integration of Stokes parameters
// Need to be given a vector of y and dydx (points along rays).
// Returns (I,Q,U,V,tau_I)
std::valarray<double>& RadiativeTransfer::IQUV_integrate(std::vector<double> [], std::vector<double> [], std::valarray<double>& iquv)
{
  return iquv;
}
};
