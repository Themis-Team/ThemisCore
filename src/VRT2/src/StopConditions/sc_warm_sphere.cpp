#include "sc_warm_sphere.h"

namespace VRT2 {
SCWarmSphere::SCWarmSphere(Metric& g, double router, double rinner, double T)
  : StopCondition(g,router,rinner), _g(g), _rout(router), _rin(rinner), _T(VRT2_Constants::k*T)
{
}

/*** Condition at which propagation stops regardless of adiabaticity ***/
int SCWarmSphere::stop_condition(double y[], double dydx[])
{
  if ( (y[1] > _rout) && (dydx[1] < 0) ) // moving into region, i.e. leaving region in - time
    return 1; // Stop
  else if (y[1] < _rin*_g.horizon())
    return 1;
  else
    return 0; // Don't Stop
}

/*** Intensity at stopped place (for optically thick stuff) ***/
double SCWarmSphere::I(double y[], int mode)
{
  if (y[1]<_rin*_g.horizon()) {
    FourVector<double> k(_g);
    k.mkcov(y+4);
    FourVector<double> u(_g);
    u.mkcov(-1,0,0,0);
    u *= 1.0/std::sqrt(-(u*u));
    
    double omega = -_omega_scale * (k*u);
    
    return 2*_T/( 4.0*VRT2_Constants::pi*VRT2_Constants::pi
		  * VRT2_Constants::c*VRT2_Constants::c
		  * omega );
  } else
    return 0.0;
}

/*** Stokes' Parameters at stopped place ***/
std::valarray<double> SCWarmSphere::IQUV(double y[])
{
  std::valarray<double> iquv(0.0,5);

  iquv[0] = I(y,0);

  return iquv;
}




SCAccretingWarmSphere::SCAccretingWarmSphere(Metric& g, double router, double rinner, double Mdot)
  : SCWarmSphere(g,router,rinner,T_from_Mdot(Mdot,rinner)), _Mdot(Mdot)
{
}

// Note that rinner is in units of horizon radii
double SCAccretingWarmSphere::T_from_Mdot(double Mdot, double rinner)
{
  Mdot *= 2.0e33/( 365.25 * 24 * 3600 );  // Put mdot from Msun/yr to g/s

  double T;
  if (rinner>1.5)
    T = std::pow( Mdot * VRT2_Constants::c*VRT2_Constants::c
		  / (4.0*VRT2_Constants::pi*VRT2_Constants::sigma 
		     * (1.0 - 1.0/rinner) * 0.25*rinner*rinner
		     * VRT2_Constants::M_SgrA_cm*VRT2_Constants::M_SgrA_cm), 0.25 );
  else
    T = std::pow( Mdot * VRT2_Constants::c*VRT2_Constants::c
		  / (4.0*VRT2_Constants::pi*VRT2_Constants::sigma 
		     * (1.0-1.0/rinner)*(1.0-1.0/rinner) * 27.0
		     * VRT2_Constants::M_SgrA_cm*VRT2_Constants::M_SgrA_cm), 0.25 );

  //std::cout << std::setw(15) << T << std::endl;
  return T;
}

void SCAccretingWarmSphere::set_inner_radius(double rinner)
{
  SCWarmSphere::set_inner_radius(rinner);
  SCWarmSphere::set_temperature(T_from_Mdot(_Mdot,rinner));
}
};
