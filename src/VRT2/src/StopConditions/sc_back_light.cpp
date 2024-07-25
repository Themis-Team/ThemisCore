#include "sc_back_light.h"

/*** Condition at which propagation stops regardless of adiabaticity ***/
namespace VRT2 {
int SC_BackLight::stop_condition(double y[], double dydx[])
{
  //if ( y[0]>0 ) // WTF!
    //return 1;

  //if (_g.ginv(0,0)<-1.0/(_rin-1.0))
  if (_g.ginv(0,0)<-30.0)
    return 1; // Alternate horizon checker

  if ( (y[1] > _rout) && (dydx[1] < 0) ) // moving into region, i.e. leaving region in - time
    return 1; // Stop
  else if (y[1] < _rin*_g.horizon())
    return 1;
  else
    return 0; // Don't Stop
}

/*** Intensity at stopped place (for optically thick stuff) ***/
double SC_BackLight::I(double y[], int mode)
{
  if (y[1]>_rout)
    return _Iinf;
  else
    return 0.0;
}

/*** Stokes' Parameters at stopped place ***/
std::valarray<double> SC_BackLight::IQUV(double y[])
{
  std::valarray<double> iquv(0.0,5);
  for (size_t i=0; i<5; ++i)
      iquv[i] = 0.0;
  iquv[0] = I(y,0);

  return iquv;
}
};
