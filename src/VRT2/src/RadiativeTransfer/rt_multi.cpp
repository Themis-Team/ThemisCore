// Include Statements
#include "rt_multi.h"

namespace VRT2 {
RT_Multi::RT_Multi(Metric& g, std::vector<RadiativeTransfer*> rts)
  : RT_RungeKutta(g), _rts(rts)
{
}

RT_Multi::RT_Multi(const double y[], Metric& g, std::vector<RadiativeTransfer*> rts)
  : RT_RungeKutta(y,g), _rts(rts)
{
}

RT_Multi::RT_Multi(FourVector<double>& x, FourVector<double>& k, Metric& g, std::vector<RadiativeTransfer*> rts)
  : RT_RungeKutta(x,k,g), _rts(rts)
{
}


void RT_Multi::IQUV_integrate_initialize(std::vector<double> y[], std::vector<double> dydx[], std::valarray<double>& iquv0)
{
  for (size_t i=0; i<_rts.size(); ++i)
    _rts[i]->IQUV_integrate_initialize(y,dydx,iquv0);
}


void RT_Multi::set_frequency_scale(double omega0)
{
  for (size_t i=0; i<_rts.size(); ++i)
    _rts[i]->set_frequency_scale(omega0);
}

void RT_Multi::set_length_scale(double L)
{
  for (size_t i=0; i<_rts.size(); ++i)
    _rts[i]->set_length_scale(L);
}

void RT_Multi::reinitialize(const double y[])
{
  RT_RungeKutta::reinitialize(y);
  for (size_t i=0; i<_rts.size(); ++i)
    _rts[i]->reinitialize(y);
}

void RT_Multi::reinitialize(FourVector<double>& x, FourVector<double>& k)
{
  RT_RungeKutta::reinitialize(x,k);
  for (size_t i=0; i<_rts.size(); ++i)
    _rts[i]->reinitialize(x,k);
}

// Characteristic RT distance
double RT_Multi::dlambda(const double y[], const double dydx[])
{
  double dlam_min = 1.0;
  for (size_t i=0; i<_rts.size(); ++i)
    dlam_min = std::min( dlam_min, _rts[i]->dlambda(y,dydx) );
  return dlam_min;
}

double RT_Multi::stable_step_size(double h, const double y[], const double dydx[])
{
  double sss_min = h;
  for (size_t i=0; i<_rts.size(); ++i)
    sss_min = std::min( sss_min, _rts[i]->dlambda(y,dydx) );
  return sss_min;
}

// Absorption vectors
std::valarray<double>& RT_Multi::IQUV_abs(const double iquv[], const double dydx[])
{
  _iquv_abs = 0.0;
  for (size_t i=0; i<_rts.size(); ++i)
    _iquv_abs += _rts[i]->IQUV_abs(iquv,dydx);
  return (_iquv_abs);
}
double RT_Multi::isotropic_absorptivity(const double dydx[])
{
  double ia = 0.0;
  for (size_t i=0; i<_rts.size(); ++i)
    ia = std::max( ia, _rts[i]->isotropic_absorptivity(dydx) );
  return ia;
}
// Emissivity vector
std::valarray<double>& RT_Multi::IQUV_ems(const double dydx[])
{
  _iquv_ems = 0.0;
  for (size_t i=0; i<_rts.size(); ++i)
    _iquv_ems += _rts[i]->IQUV_ems(dydx);
  return (_iquv_ems);
}

// Linear RT Change
void RT_Multi::IQUV_rotate(double iquv[], double xi, const double yi[], const double dydxi[], double xf, const double yf[], const double dydxf[])
{
  for (size_t i=0; i<_rts.size(); ++i)
    _rts[i]->IQUV_rotate(iquv,xi,yi,dydxi,xf,yf,dydxf);
}








void RT_Multi::dump_ray(std::string fname)
{
  std::ofstream rout(fname.c_str(),std::ios_base::app);

  size_t N = _ya[0].size();
  double y[8], dydx[8];
  double dlambda = (_ya[0][0]-_ya[0][N-1])/double(2*N+1);
  double lambda;
  for (int i=0; i<2*int(N); ++i)
  {
    lambda = _ya[0][N-1] + dlambda*(i+1);
    interp(lambda,y,dydx);
    _g.reset(y);
    reinitialize(y);

    rout << std::setw(15) << lambda
	 << std::setw(15) << y[0]
	 << std::setw(15) << y[1]
	 << std::setw(15) << y[2]
	 << std::setw(15) << y[3]
	 << std::setw(15) << y[4]
	 << std::setw(15) << y[5]
	 << std::setw(15) << y[6]
	 << std::setw(15) << y[7];
    for (size_t i=0; i<_rts.size(); ++i)
      _rts[i]->dump(rout,dydx);

    rout << std::endl;
  }
  rout << "\n" << std::endl;

  std::cerr << "Output Ray\n";

  //std::exit(1);
}
};
